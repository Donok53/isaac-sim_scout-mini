#!/usr/bin/env python3
# collect_nav2_dataset.py
# Nav2 teacher 기반 자동 에피소드 데이터 수집기
#
# 저장 구조:
# data/nav2_dataset_YYYYmmdd_HHMMSS/
#   ep_000001/
#     images/img_000001.jpg ...
#     samples.jsonl
#     meta.json
#   ep_000002/ ...
#
# 요구:
# - /navigate_to_pose (nav2_msgs/action/NavigateToPose) action server
# - /cmd_vel (Twist) : Nav2가 publish하는 teacher action
# - /odom, camera, lidar topics

import os
import json
import time
import math
import random
import struct
from datetime import datetime

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient

from geometry_msgs.msg import Twist, PoseStamped, PoseWithCovarianceStamped
from nav_msgs.msg import Odometry, OccupancyGrid
from sensor_msgs.msg import PointCloud2, Image
from nav2_msgs.action import NavigateToPose

import cv2
from cv_bridge import CvBridge


def clamp(x, lo, hi):
    return lo if x < lo else hi if x > hi else x


def wrap_pi(a):
    return math.atan2(math.sin(a), math.cos(a))


def quat_to_yaw(x, y, z, w):
    t3 = 2.0 * (w * z + x * y)
    t4 = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(t3, t4)


def yaw_to_quat(yaw):
    # (0,0,yaw)
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    return (0.0, 0.0, sy, cy)


class Nav2DatasetCollector(Node):
    def __init__(self):
        super().__init__("nav2_dataset_collector")
        self.bridge = CvBridge()

        # ----------------------------
        # ENV / Params
        # ----------------------------
        self.base_dir = os.getenv("BASE_DIR", "data")

        self.episodes = int(os.getenv("EPISODES", "50"))
        self.episode_timeout = float(os.getenv("EP_TIMEOUT_SEC", "90.0"))
        self.warmup_sec = float(os.getenv("WARMUP_SEC", "1.0"))
        self.sample_hz = float(os.getenv("SAMPLE_HZ", "10.0"))
        self.max_staleness = float(os.getenv("MAX_STALENESS_SEC", "0.7"))

        self.frame_id = os.getenv("FRAME_ID", "odom")  # 네 기존 코드가 odom 기준이라 기본 odom
        self.nav_action_name = os.getenv("NAV_ACTION", "/navigate_to_pose")

        self.camera_topic = os.getenv("CAMERA_TOPIC", "/camera/color/image_raw")
        self.lidar_topic = os.getenv("LIDAR_TOPIC", "/ouster/points")
        self.odom_topic = os.getenv("ODOM_TOPIC", "/odom")
        self.cmd_topic = os.getenv("CMD_TOPIC", "/cmd_vel")
        self.map_topic = os.getenv("MAP_TOPIC", "/map")  # 있으면 goal 샘플링에 사용

        self.goal_min_dist = float(os.getenv("GOAL_MIN_DIST", "6.0"))
        self.goal_max_dist = float(os.getenv("GOAL_MAX_DIST", "20.0"))
        self.goal_reach_dist = float(os.getenv("GOAL_REACH_DIST", "0.8"))

        # /map 없을 때 bounding box 샘플링(odom 좌표계 가정)
        # 형식: "xmin,xmax,ymin,ymax"
        bounds = os.getenv("GOAL_BOUNDS", "-3.0,3.0,-3.0,3.0")
        bx = [float(x.strip()) for x in bounds.split(",")]
        self.bxmin, self.bxmax, self.bymin, self.bymax = bx[0], bx[1], bx[2], bx[3]

        # stuck 판정(선택)
        self.stuck_check_sec = float(os.getenv("STUCK_CHECK_SEC", "5.0"))
        self.stuck_move_min = float(os.getenv("STUCK_MOVE_MIN", "0.15"))  # 이 이하 이동이면 stuck
        self.enable_stuck = int(os.getenv("ENABLE_STUCK_CHECK", "1")) == 1

        # 실패 에피소드 저장 여부
        # 0: 실패도 저장, 1: 실패는 폴더 삭제(권장 X 디버깅 불가)
        self.discard_fail = int(os.getenv("DISCARD_FAIL", "0")) == 1

        # 라이다를 181 bins로 축약 저장(네 코드 스타일 유지)
        self.fov_deg = float(os.getenv("FOV_DEG", "180.0"))
        self.n_bins = int(os.getenv("N_BINS", "181"))
        self.max_range = float(os.getenv("MAX_RANGE", "6.0"))
        self.ang_min = -math.radians(self.fov_deg / 2.0)
        self.ang_max = +math.radians(self.fov_deg / 2.0)
        self.ang_res = (self.ang_max - self.ang_min) / (self.n_bins - 1)
        self.point_stride = int(os.getenv("POINT_STRIDE", "10"))  # pointcloud subsample

        # ----------------------------
        # Session directories
        # ----------------------------
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_dir = os.path.join(self.base_dir, f"nav2_dataset_{ts}")
        os.makedirs(self.session_dir, exist_ok=True)

        self.get_logger().info(f"📁 Session: {self.session_dir}")
        self.get_logger().info(f"Topics: camera={self.camera_topic}, lidar={self.lidar_topic}, odom={self.odom_topic}, cmd={self.cmd_topic}")
        self.get_logger().info(f"Nav2 action: {self.nav_action_name} | frame_id={self.frame_id}")

        # ----------------------------
        # Latest data buffers
        # ----------------------------
        self.last_odom = None
        self.last_odom_t = None

        self.last_cmd = None
        self.last_cmd_t = None

        self.last_lidar = None
        self.last_lidar_t = None

        self.last_image_path = None
        self.last_image_t = None

        self.map_msg = None

        # Robot state (for convenience)
        self.x = self.y = self.z = 0.0
        self.yaw = 0.0

        # Distance tracking (episode-level)
        self._prev_xy = None
        self._ep_travel = 0.0
        self._stuck_ref_time = None
        self._stuck_ref_dist = 0.0

        # ----------------------------
        # ROS interfaces
        # ----------------------------
        self.create_subscription(Odometry, self.odom_topic, self.odom_cb, 50)
        self.create_subscription(Twist, self.cmd_topic, self.cmd_cb, 50)
        self.create_subscription(PointCloud2, self.lidar_topic, self.lidar_cb, 10)
        self.create_subscription(Image, self.camera_topic, self.camera_cb, 10)
        self.create_subscription(OccupancyGrid, self.map_topic, self.map_cb, 1)

        self.initpose_pub = self.create_publisher(PoseWithCovarianceStamped, "/initialpose", 10)
        self.stop_pub = self.create_publisher(Twist, self.cmd_topic, 10)  # 에피소드 종료 시 정지(같은 토픽으로 0만 1회 publish)

        self.nav_client = ActionClient(self, NavigateToPose, self.nav_action_name)

        # ----------------------------
        # Episode state machine
        # ----------------------------
        self.ep_id = 0
        self.ep_active = False
        self.ep_dir = None
        self.images_dir = None
        self.samples_fp = None
        self.sample_idx = 0

        self.start_pose = None
        self.goal_pose = None
        self.goal_handle = None
        self.result_future = None

        self.ep_start_wall = None  # time.time()
        self.ep_start_ros = None   # node clock time

        # Timers
        self.episode_timer = self.create_timer(0.2, self.episode_loop)
        self.sample_timer = self.create_timer(1.0 / max(1e-6, self.sample_hz), self.sample_tick)

    # ----------------------------
    # Callbacks
    # ----------------------------
    def map_cb(self, msg: OccupancyGrid):
        self.map_msg = msg

    def odom_cb(self, msg: Odometry):
        self.last_odom = msg
        self.last_odom_t = self._stamp_to_float(msg.header.stamp)

        self.x = float(msg.pose.pose.position.x)
        self.y = float(msg.pose.pose.position.y)
        self.z = float(msg.pose.pose.position.z)

        q = msg.pose.pose.orientation
        self.yaw = float(quat_to_yaw(q.x, q.y, q.z, q.w))

        # travel update (only during episode)
        if self.ep_active:
            if self._prev_xy is None:
                self._prev_xy = (self.x, self.y)
            else:
                dx = self.x - self._prev_xy[0]
                dy = self.y - self._prev_xy[1]
                self._ep_travel += math.hypot(dx, dy)
                self._prev_xy = (self.x, self.y)

    def cmd_cb(self, msg: Twist):
        self.last_cmd = msg
        self.last_cmd_t = self._now_float()

    def camera_cb(self, msg: Image):
        if not self.ep_active:
            return

        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception:
            return

        self.sample_idx += 1
        img_name = f"img_{self.sample_idx:06d}.jpg"
        img_path = os.path.join(self.images_dir, img_name)

        try:
            cv2.imwrite(img_path, cv_image)
        except Exception:
            return

        self.last_image_path = os.path.join("images", img_name)
        self.last_image_t = self._stamp_to_float(msg.header.stamp)

    def lidar_cb(self, msg: PointCloud2):
        if not self.ep_active:
            return

        ranges = [self.max_range] * self.n_bins

        try:
            step = msg.point_step
            data = msg.data

            # subsample
            stride = step * max(1, self.point_stride)

            for off in range(0, len(data), stride):
                if off + 12 > len(data):
                    break

                x = struct.unpack_from("f", data, off + 0)[0]
                y = struct.unpack_from("f", data, off + 4)[0]
                z = struct.unpack_from("f", data, off + 8)[0]

                # 전방 / 지면 근처 포인트만 (너 코드 스타일 유지)
                if not (0.05 < x < self.max_range and abs(y) < 3.0 and -0.45 < z < 0.15):
                    continue

                r = math.hypot(x, y)
                if r < 0.03:
                    continue

                ang = math.atan2(y, x)
                if ang < self.ang_min or ang > self.ang_max:
                    continue

                idx = int((ang - self.ang_min) / self.ang_res)
                if 0 <= idx < self.n_bins and r < ranges[idx]:
                    ranges[idx] = r

        except Exception:
            ranges = [self.max_range] * self.n_bins

        self.last_lidar = {
            "timestamp": self._stamp_to_float(msg.header.stamp),
            "ranges": ranges,
            "min_distance": float(min(ranges)),
        }
        self.last_lidar_t = self.last_lidar["timestamp"]

    # ----------------------------
    # Episode logic
    # ----------------------------
    def episode_loop(self):
        # 0) Nav2 action server 대기
        if not self.nav_client.server_is_ready():
            self.get_logger().info("⏳ Waiting Nav2 action server...", throttle_duration_sec=3.0)
            return

        # 1) 센서 warmup: odom 하나라도 와야 시작
        if self.last_odom is None:
            self.get_logger().info("⏳ Waiting /odom...", throttle_duration_sec=3.0)
            return

        # 2) 에피소드 진행 중이면 종료 체크
        if self.ep_active:
            self._check_episode_done()
            return

        # 3) 다음 에피소드 시작
        if self.ep_id >= self.episodes:
            self.get_logger().info("✅ All episodes done. Shutting down.")
            self._stop_robot()
            rclpy.shutdown()
            return

        self._start_episode()

    def _start_episode(self):
        self.ep_id += 1

        # warmup delay
        time.sleep(max(0.0, self.warmup_sec))

        # create ep folder
        self.ep_dir = os.path.join(self.session_dir, f"ep_{self.ep_id:06d}")
        self.images_dir = os.path.join(self.ep_dir, "images")
        os.makedirs(self.images_dir, exist_ok=True)

        # open samples.jsonl
        self.samples_fp = open(os.path.join(self.ep_dir, "samples.jsonl"), "w", buffering=1)

        # reset counters
        self.sample_idx = 0
        self._ep_travel = 0.0
        self._prev_xy = (self.x, self.y)
        self._stuck_ref_time = self._now_float()
        self._stuck_ref_dist = self._ep_travel

        # start pose = 현재 odom
        self.start_pose = {
            "x": float(self.x),
            "y": float(self.y),
            "z": float(self.z),
            "yaw": float(self.yaw),
        }

        # initialpose publish (Nav2 localization 쓰는 경우 도움)
        self._publish_initialpose(self.start_pose["x"], self.start_pose["y"], self.start_pose["yaw"])

        # goal sample
        goal_x, goal_y, goal_yaw = self._sample_goal()

        self.goal_pose = {"x": goal_x, "y": goal_y, "z": float(self.z), "yaw": goal_yaw}

        # send goal
        goal_msg = NavigateToPose.Goal()
        goal_msg.pose = self._make_pose_stamped(goal_x, goal_y, goal_yaw, frame_id=self.frame_id)

        self.ep_start_wall = time.time()
        self.ep_start_ros = self.get_clock().now()

        send_future = self.nav_client.send_goal_async(goal_msg)
        send_future.add_done_callback(self._on_goal_response)

        self.ep_active = True

        self.get_logger().info(
            f"🚀 EP {self.ep_id}/{self.episodes} start=({self.start_pose['x']:.2f},{self.start_pose['y']:.2f}) "
            f"goal=({goal_x:.2f},{goal_y:.2f}) timeout={self.episode_timeout:.0f}s"
        )

        # write meta early
        meta = {
            "episode_id": self.ep_id,
            "frame_id": self.frame_id,
            "start": self.start_pose,
            "goal": self.goal_pose,
            "result": None,
            "failure_reason": None,
            "created_at": datetime.now().isoformat(),
        }
        with open(os.path.join(self.ep_dir, "meta.json"), "w") as f:
            json.dump(meta, f, indent=2)

    def _on_goal_response(self, future):
        try:
            goal_handle = future.result()
        except Exception as e:
            self.get_logger().error(f"Goal send failed: {e}")
            self._end_episode("failure", "send_goal_exception")
            return

        if not goal_handle.accepted:
            self.get_logger().warn("❌ Goal rejected by Nav2")
            self._end_episode("failure", "goal_rejected")
            return

        self.goal_handle = goal_handle
        self.result_future = goal_handle.get_result_async()

    def _check_episode_done(self):
        # timeout
        if (time.time() - self.ep_start_wall) > self.episode_timeout:
            self.get_logger().warn("⏰ timeout -> cancel goal")
            self._cancel_goal_and_end("failure", "timeout")
            return

        # stuck check
        if self.enable_stuck and self._is_stuck():
            self.get_logger().warn("🧱 stuck -> cancel goal")
            self._cancel_goal_and_end("failure", "stuck")
            return

        # goal reached quick check (거리 기반)
        gx, gy = self.goal_pose["x"], self.goal_pose["y"]
        if math.hypot(gx - self.x, gy - self.y) < self.goal_reach_dist:
            # Nav2 결과 기다리지 않고도 성공으로 처리 가능(일단은)
            self.get_logger().info("🎯 reach_dist met")
            self._end_episode("success", None)
            return

        # nav2 result
        if self.result_future is not None and self.result_future.done():
            try:
                res = self.result_future.result()
                status = int(res.status)
            except Exception as e:
                self.get_logger().error(f"Result read failed: {e}")
                self._end_episode("failure", "result_exception")
                return

            # nav2 status codes (GoalStatus)
            # 4: SUCCEEDED, 5: CANCELED, 6: ABORTED (일반적)
            if status == 4:
                self._end_episode("success", None)
            elif status == 5:
                self._end_episode("failure", "nav2_canceled")
            else:
                self._end_episode("failure", f"nav2_status_{status}")

    def _cancel_goal_and_end(self, result, reason):
        if self.goal_handle is not None:
            cancel_future = self.goal_handle.cancel_goal_async()
            # cancel 완료를 기다리지 않고 종료 처리
            _ = cancel_future
        self._end_episode(result, reason)

    def _end_episode(self, result, reason):
        # stop robot (one-shot)
        self._stop_robot()

        # close samples file
        try:
            if self.samples_fp is not None:
                self.samples_fp.flush()
                self.samples_fp.close()
        except Exception:
            pass

        # finalize meta
        meta_path = os.path.join(self.ep_dir, "meta.json")
        try:
            with open(meta_path, "r") as f:
                meta = json.load(f)
        except Exception:
            meta = {}

        meta["result"] = result
        meta["failure_reason"] = reason
        meta["duration_sec"] = round(time.time() - self.ep_start_wall, 3) if self.ep_start_wall else None
        meta["num_images"] = int(self.sample_idx)
        meta["travel_m"] = round(float(self._ep_travel), 3)

        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

        self.get_logger().info(
            f"✅ EP {self.ep_id} done: {result}"
            + (f" ({reason})" if reason else "")
            + f" | travel={self._ep_travel:.2f}m | imgs={self.sample_idx}"
        )

        # optionally discard fail
        if self.discard_fail and result != "success":
            try:
                # 위험하지만 옵션으로만 제공
                for root, dirs, files in os.walk(self.ep_dir, topdown=False):
                    for name in files:
                        os.remove(os.path.join(root, name))
                    for name in dirs:
                        os.rmdir(os.path.join(root, name))
                os.rmdir(self.ep_dir)
                self.get_logger().info(f"🗑️ Discarded failed episode folder: {self.ep_dir}")
            except Exception as e:
                self.get_logger().warn(f"Failed to discard folder: {e}")

        # reset episode state
        self.ep_active = False
        self.goal_handle = None
        self.result_future = None
        self.samples_fp = None
        self.last_image_path = None
        self.last_image_t = None
        self.last_lidar = None
        self.last_lidar_t = None
        self._prev_xy = None

    # ----------------------------
    # Sampling / Writing
    # ----------------------------
    def sample_tick(self):
        if not self.ep_active:
            return

        now = self._now_float()

        # availability check
        if self.last_cmd is None or self.last_odom is None or self.last_lidar is None or self.last_image_path is None:
            return

        # staleness check
        if self.last_cmd_t is None or self.last_odom_t is None or self.last_lidar_t is None or self.last_image_t is None:
            return

        if (now - self.last_cmd_t) > self.max_staleness:
            return
        if abs(now - self.last_odom_t) > self.max_staleness:
            return
        if abs(now - self.last_lidar_t) > self.max_staleness:
            return
        if abs(now - self.last_image_t) > self.max_staleness:
            return

        # pack sample (jsonl)
        sample = {
            "t": now,
            "episode_id": self.ep_id,
            "camera": {
                "image_path": self.last_image_path,
                "timestamp": self.last_image_t,
            },
            "lidar": {
                "timestamp": self.last_lidar["timestamp"],
                "min_distance": self.last_lidar["min_distance"],
                "ranges": self.last_lidar["ranges"],  # 181 bins
            },
            "odometry": {
                "x": self.x, "y": self.y, "z": self.z, "yaw": self.yaw
            },
            "action": {
                "linear_x": float(self.last_cmd.linear.x),
                "angular_z": float(self.last_cmd.angular.z),
            }
        }

        try:
            self.samples_fp.write(json.dumps(sample) + "\n")
        except Exception:
            pass

    def _sample_goal(self):
        sx, sy = self.start_pose["x"], self.start_pose["y"]

        # 1) map 기반 샘플링(가능하면)
        if self.map_msg is not None and self.frame_id == "map":
            for _ in range(2000):
                gx, gy = self._sample_free_from_map(self.map_msg)
                if math.hypot(gx - sx, gy - sy) < self.goal_min_dist:
                    continue
                if math.hypot(gx - sx, gy - sy) > self.goal_max_dist:
                    continue
                yaw = random.uniform(-math.pi, math.pi)
                return gx, gy, yaw

        # 2) fallback: bounding box + min/max dist
        for _ in range(2000):
            gx = random.uniform(self.bxmin, self.bxmax)
            gy = random.uniform(self.bymin, self.bymax)
            d = math.hypot(gx - sx, gy - sy)
            if d < self.goal_min_dist or d > self.goal_max_dist:
                continue
            yaw = random.uniform(-math.pi, math.pi)
            return gx, gy, yaw

        # fallback worst-case
        yaw = random.uniform(-math.pi, math.pi)
        return (sx + self.goal_min_dist, sy, yaw)

    def _sample_free_from_map(self, m: OccupancyGrid):
        w, h = m.info.width, m.info.height
        res = m.info.resolution
        ox, oy = m.info.origin.position.x, m.info.origin.position.y

        for _ in range(5000):
            idx = random.randrange(w * h)
            if m.data[idx] == 0:
                cx = idx % w
                cy = idx // w
                x = ox + (cx + 0.5) * res
                y = oy + (cy + 0.5) * res
                return x, y

        return (ox, oy)

    def _publish_initialpose(self, x, y, yaw):
        msg = PoseWithCovarianceStamped()
        msg.header.frame_id = self.frame_id
        msg.header.stamp = self.get_clock().now().to_msg()

        msg.pose.pose.position.x = float(x)
        msg.pose.pose.position.y = float(y)
        qx, qy, qz, qw = yaw_to_quat(yaw)
        msg.pose.pose.orientation.x = qx
        msg.pose.pose.orientation.y = qy
        msg.pose.pose.orientation.z = qz
        msg.pose.pose.orientation.w = qw

        # 약간의 확신 있는 초기값(필요하면 조정)
        msg.pose.covariance[0] = 0.05
        msg.pose.covariance[7] = 0.05
        msg.pose.covariance[35] = 0.1

        self.initpose_pub.publish(msg)

    def _make_pose_stamped(self, x, y, yaw, frame_id):
        ps = PoseStamped()
        ps.header.frame_id = frame_id
        ps.header.stamp = self.get_clock().now().to_msg()
        ps.pose.position.x = float(x)
        ps.pose.position.y = float(y)
        ps.pose.position.z = 0.0
        qx, qy, qz, qw = yaw_to_quat(yaw)
        ps.pose.orientation.x = qx
        ps.pose.orientation.y = qy
        ps.pose.orientation.z = qz
        ps.pose.orientation.w = qw
        return ps

    def _is_stuck(self):
        now = self._now_float()
        if self._stuck_ref_time is None:
            self._stuck_ref_time = now
            self._stuck_ref_dist = self._ep_travel
            return False

        if (now - self._stuck_ref_time) < self.stuck_check_sec:
            return False

        moved = self._ep_travel - self._stuck_ref_dist
        self._stuck_ref_time = now
        self._stuck_ref_dist = self._ep_travel

        # 목표가 가까우면 stuck 판정 안 함
        gx, gy = self.goal_pose["x"], self.goal_pose["y"]
        if math.hypot(gx - self.x, gy - self.y) < (self.goal_reach_dist * 2.0):
            return False

        return moved < self.stuck_move_min

    def _stop_robot(self):
        msg = Twist()
        msg.linear.x = 0.0
        msg.angular.z = 0.0
        try:
            self.stop_pub.publish(msg)
        except Exception:
            pass

    def _stamp_to_float(self, stamp):
        return float(stamp.sec) + float(stamp.nanosec) * 1e-9

    def _now_float(self):
        # node clock time (wall-like enough)
        return self.get_clock().now().nanoseconds / 1e9


def main():
    rclpy.init()
    node = Nav2DatasetCollector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            node._stop_robot()
        except Exception:
            pass
        try:
            node.destroy_node()
        except Exception:
            pass
        try:
            if rclpy.ok():
                rclpy.shutdown()
        except Exception:
            pass


if __name__ == "__main__":
    main()
