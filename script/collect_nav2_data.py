#!/usr/bin/env python3
# collect_nav2_data.py - Path-Faithful Avoidance (Early detect + Fast rejoin)

import rclpy
from rclpy.node import Node

from geometry_msgs.msg import Twist, PoseStamped, Point
from sensor_msgs.msg import PointCloud2, Image
from nav_msgs.msg import Odometry, Path
from visualization_msgs.msg import Marker, MarkerArray

import json
import cv2
from cv_bridge import CvBridge
from datetime import datetime
import struct
import math
import os


def clamp(x, lo, hi):
    return lo if x < lo else hi if x > hi else x


def wrap_pi(a):
    return math.atan2(math.sin(a), math.cos(a))


class Nav2DataCollector(Node):
    def __init__(self):
        super().__init__('nav2_data_collector')

        # -------------------------
        # Data storage
        # -------------------------
        self.data_buffer = []
        self.current_sample = {}
        self.bridge = CvBridge()

        # -------------------------
        # Robot state
        # -------------------------
        self.current_x = 0.0
        self.current_y = 0.0
        self.current_z = 0.0
        self.current_yaw = 0.0

        # Sim check
        self.odom_received = False
        self.simulation_running = False
        self.last_odom_time = None

        # Progress tracking (실제 주행 거리)
        self.start_x = None
        self.start_y = None
        self.total_distance_traveled = 0.0
        self.last_x = None
        self.last_y = None
        self.min_distance_to_complete = 30.0  # 최소 30m 주행해야 완료

        # -------------------------
        # Path following (No waypoints)
        # -------------------------
        self.lookahead_distance = 0.6  # 코너 컷 완화
        self.global_path = []
        self.path_generated = False
        self.path_completed = False

        # progress index (점프 방지)
        self.progress_idx = 0
        self.closest_search_window = 20  # path 생성 후 설정

        # -------------------------
        # Robot size (Scout-mini approx)
        # -------------------------
        self.ROBOT_WIDTH = 0.55
        self.ROBOT_RADIUS = self.ROBOT_WIDTH * 0.5
        self.SAFETY_MARGIN = 0.20
        self.BUBBLE_RADIUS = self.ROBOT_RADIUS + self.SAFETY_MARGIN

        # -------------------------
        # Control
        # -------------------------
        self.max_v = 0.6
        self.max_w = 2.5

        # smoothing
        self.alpha_v = 0.35
        self.alpha_w = 0.70
        self.cmd_filt_v = 0.0
        self.cmd_filt_w = 0.0

        self.current_action = {'linear_x': 0.0, 'angular_z': 0.0}

        # -------------------------
        # Follow-the-Gap polar scan config
        # -------------------------
        self.FOV_DEG = 180.0
        self.ANG_MIN = -math.radians(self.FOV_DEG / 2.0)
        self.ANG_MAX = +math.radians(self.FOV_DEG / 2.0)
        self.N_BINS = 181
        self.ANG_RES = (self.ANG_MAX - self.ANG_MIN) / (self.N_BINS - 1)

        self.MAX_RANGE = 6.0
        self.ranges = [self.MAX_RANGE] * self.N_BINS

        # Obstacle summary
        self.min_obstacle_distance = 999.0
        self.obstacle_angle = 0.0
        self.front_min_dist = 999.0
        self.path_clear_dist = 999.0  # 경로 진행방향 기준 clearance

        # -------------------------
        # 핵심 튜닝(요구사항 반영)
        # -------------------------
        # (1) 회피/복귀는 path_clear 중심으로!
        self.AVOID_START_PATH = 2.6   # path 진행방향이 이보다 막히면 회피 시작
        self.AVOID_CLEAR_PATH = 3.1   # path 진행방향이 이보다 뚫리면 즉시 TRACK 복귀

        # front는 "충돌 임박"만 담당
        self.EMERGENCY_DIST = 0.40

        # 갭 free 판정
        self.GAP_CLEAR_DIST = 1.35

        # (2) 회피 시 경로에서 너무 벗어나지 않도록 조향 편차 캡
        self.MAX_AVOID_DELTA_DEG = 25.0  # desired(경로) 대비 회피각 최대 편차
        self.prev_gap_angle = 0.0        # 히스테리시스(좌우 흔들림 감소)

        # Path deviation soft limit
        self.MAX_PATH_DEVIATION = 1.0

        # Mode
        self.mode = "TRACK"  # TRACK / GAP_AVOID / EMERGENCY / ESCAPE

        # Stuck escape (후진은 여기서만)
        self.last_progress_check_time = self.get_clock().now()
        self.last_progress_distance = 0.0
        self.escape_until = None
        self.escape_turn_dir = 1.0

        # -------------------------
        # ROS interfaces
        # -------------------------
        self.lidar_sub = self.create_subscription(
            PointCloud2, '/ouster/points', self.lidar_callback, 10
        )
        self.camera_sub = self.create_subscription(
            Image, '/camera/color/image_raw', self.camera_callback, 10
        )
        self.odom_sub = self.create_subscription(
            Odometry, '/odom', self.odom_callback, 10
        )

        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.path_pub = self.create_publisher(Path, '/planned_path', 10)
        self.marker_pub = self.create_publisher(MarkerArray, '/path_markers', 10)

        # -------------------------
        # Save dirs
        # -------------------------
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        session_dir = f"nav2_data_{timestamp}"

        self.base_dir = "data"
        self.session_dir = os.path.join(self.base_dir, session_dir)
        self.images_dir = os.path.join(self.session_dir, "images")

        os.makedirs(self.base_dir, exist_ok=True)
        os.makedirs(self.session_dir, exist_ok=True)
        os.makedirs(self.images_dir, exist_ok=True)

        self.sample_count = 0
        self.get_logger().info(f"📁 {self.session_dir}/")
        self.get_logger().info("🛤️  Waiting for robot position to generate path...")

        # -------------------------
        # Timers
        # -------------------------
        self.control_timer = self.create_timer(0.05, self.control_loop)
        self.sim_check_timer = self.create_timer(2.0, self.check_simulation_status)
        self.marker_timer = self.create_timer(0.5, self.publish_path_markers)
        self.progress_timer = self.create_timer(2.0, self.check_completion)

    # ============================================================
    # Path generation (원본 유지)
    # ============================================================
    def generate_path_from_start(self, start_x, start_y):
        path = []
        self.get_logger().info(f"🚀 Generating path from ({start_x:.2f}, {start_y:.2f})")
        path.append((start_x, start_y))

        for i in range(1, 11):
            x = start_x + i * 0.3
            if x <= 3.0:
                path.append((x, start_y))

        current_x = path[-1][0]
        for i in range(1, 11):
            y = start_y + i * 0.3
            if y <= 3.0:
                path.append((current_x, y))

        current_y = path[-1][1]
        for i in range(1, 21):
            x = current_x - i * 0.3
            if x >= -3.0:
                path.append((x, current_y))

        current_x = path[-1][0]
        for i in range(1, 21):
            y = current_y - i * 0.3
            if y >= -3.0:
                path.append((current_x, y))

        current_y = path[-1][1]
        for i in range(1, 21):
            x = current_x + i * 0.3
            if x <= 3.0:
                path.append((x, current_y))

        current_x = path[-1][0]
        steps = int((0.0 - current_y) / 0.3) + 1
        for i in range(1, steps):
            y = current_y + i * 0.3
            if y <= 0.1:
                path.append((current_x, y))

        if current_x < 1.5:
            steps = int((1.5 - current_x) / 0.3) + 1
            for i in range(1, steps):
                x = current_x + i * 0.3
                if x <= 1.5:
                    path.append((x, 0.0))

        for i in range(1, 6):
            path.append((1.5, i * 0.3))
        for i in range(1, 11):
            path.append((1.5 - i * 0.3, 1.5))
        for i in range(1, 11):
            path.append((-1.5, 1.5 - i * 0.3))
        for i in range(1, 11):
            path.append((-1.5 + i * 0.3, -1.5))
        for i in range(1, 6):
            path.append((1.5, -1.5 + i * 0.3))

        for i in range(1, 6):
            x = 1.5 - i * 0.3
            path.append((x, start_y))

        path.append((start_x, start_y))
        self.get_logger().info(f"✅ Path generated: {len(path)} points")
        return path

    @staticmethod
    def quaternion_to_yaw(x, y, z, w):
        t3 = 2.0 * (w * z + x * y)
        t4 = 1.0 - 2.0 * (y * y + z * z)
        return math.atan2(t3, t4)

    # ============================================================
    # Path publish / markers
    # ============================================================
    def publish_path(self):
        if not self.path_generated or len(self.global_path) == 0:
            return
        z = float(self.current_z)

        path_msg = Path()
        path_msg.header.frame_id = "odom"
        path_msg.header.stamp = self.get_clock().now().to_msg()

        for x, y in self.global_path:
            pose = PoseStamped()
            pose.header.frame_id = "odom"
            pose.pose.position.x = float(x)
            pose.pose.position.y = float(y)
            pose.pose.position.z = float(z)
            pose.pose.orientation.w = 1.0
            path_msg.poses.append(pose)

        self.path_pub.publish(path_msg)

    def publish_path_markers(self):
        if not self.path_generated or len(self.global_path) == 0:
            return

        z = float(self.current_z)
        now = self.get_clock().now().to_msg()

        marker_array = MarkerArray()
        delete_marker = Marker()
        delete_marker.header.frame_id = "odom"
        delete_marker.action = Marker.DELETEALL
        marker_array.markers.append(delete_marker)

        line = Marker()
        line.header.frame_id = "odom"
        line.header.stamp = now
        line.ns = "path"
        line.id = 0
        line.type = Marker.LINE_STRIP
        line.action = Marker.ADD
        line.pose.orientation.w = 1.0
        line.scale.x = 0.05
        line.color.r = 0.0
        line.color.g = 0.5
        line.color.b = 1.0
        line.color.a = 0.9

        for x, y in self.global_path:
            p = Point()
            p.x = float(x)
            p.y = float(y)
            p.z = float(z)
            line.points.append(p)
        marker_array.markers.append(line)

        robot = Marker()
        robot.header.frame_id = "odom"
        robot.header.stamp = now
        robot.ns = "robot"
        robot.id = 1
        robot.type = Marker.SPHERE
        robot.action = Marker.ADD
        robot.pose.position.x = float(self.current_x)
        robot.pose.position.y = float(self.current_y)
        robot.pose.position.z = float(z)
        robot.pose.orientation.w = 1.0
        robot.scale.x = 0.25
        robot.scale.y = 0.25
        robot.scale.z = 0.25
        robot.color.r = 0.0
        robot.color.g = 1.0
        robot.color.b = 0.0
        robot.color.a = 1.0
        marker_array.markers.append(robot)

        text = Marker()
        text.header.frame_id = "odom"
        text.header.stamp = now
        text.ns = "mode"
        text.id = 2
        text.type = Marker.TEXT_VIEW_FACING
        text.action = Marker.ADD
        text.pose.position.x = float(self.current_x)
        text.pose.position.y = float(self.current_y)
        text.pose.position.z = float(z + 0.6)
        text.pose.orientation.w = 1.0
        text.scale.z = 0.25
        text.color.r = 1.0
        text.color.g = 1.0
        text.color.b = 1.0
        text.color.a = 1.0
        text.text = f"{self.mode} | front={self.front_min_dist:.2f}m | path={self.path_clear_dist:.2f}m"
        marker_array.markers.append(text)

        self.marker_pub.publish(marker_array)

    # ============================================================
    # Simulation status
    # ============================================================
    def check_simulation_status(self):
        now = self.get_clock().now()

        if not self.odom_received:
            self.get_logger().warn("⏸️  No odom", throttle_duration_sec=5.0)
            self.simulation_running = False
            return

        if self.last_odom_time is None:
            return

        dt = (now - self.last_odom_time).nanoseconds / 1e9
        if dt > 2.0:
            if self.simulation_running:
                self.get_logger().warn("⏸️  Stopped!")
            self.simulation_running = False
        else:
            if not self.simulation_running:
                self.get_logger().info("▶️  Running!")
            self.simulation_running = True

    # ============================================================
    # Completion check
    # ============================================================
    def check_completion(self):
        if not self.simulation_running or self.path_completed or not self.path_generated:
            return

        closest_idx, closest_dist = self.find_closest_point_on_path_windowed()
        progress = closest_idx / max(1, len(self.global_path) - 1)

        end_x, end_y = self.global_path[-1]
        dist_to_end = math.hypot(end_x - self.current_x, end_y - self.current_y)

        self.get_logger().info(
            f"Progress: {progress*100:.1f}% | idx={closest_idx}/{len(self.global_path)} | "
            f"closest_d={closest_dist:.2f}m | traveled={self.total_distance_traveled:.1f}m | "
            f"dist_to_end={dist_to_end:.2f}m | mode={self.mode}"
        )

        if (self.total_distance_traveled > self.min_distance_to_complete and
                progress > 0.90 and
                dist_to_end < 0.6):
            self.path_completed = True
            self.get_logger().info(
                f"🎉 Completed! {self.sample_count} samples, {self.total_distance_traveled:.1f}m traveled"
            )
            self.save_to_file()
            self.cmd_vel_pub.publish(Twist())
            raise SystemExit

    # ============================================================
    # Path utilities
    # ============================================================
    def seg_heading(self, i):
        n = len(self.global_path)
        if n < 2:
            return self.current_yaw
        if i >= n - 1:
            x0, y0 = self.global_path[n - 2]
            x1, y1 = self.global_path[n - 1]
        else:
            x0, y0 = self.global_path[i]
            x1, y1 = self.global_path[i + 1]
        return math.atan2(y1 - y0, x1 - x0)

    def signed_cross_track(self, i):
        n = len(self.global_path)
        if n < 2:
            return 0.0
        if i >= n - 1:
            i = n - 2
        x0, y0 = self.global_path[i]
        x1, y1 = self.global_path[i + 1]
        dx = x1 - x0
        dy = y1 - y0
        L = math.hypot(dx, dy)
        if L < 1e-6:
            return 0.0
        dx /= L
        dy /= L

        rx = self.current_x - x0
        ry = self.current_y - y0

        # left normal = (-dy, dx)
        cte = (-dy) * rx + (dx) * ry
        return cte

    def find_closest_point_on_path_windowed(self):
        n = len(self.global_path)
        if n == 0:
            return 0, 1e9

        back = max(4, int(self.closest_search_window * 0.25))
        fwd = self.closest_search_window

        lo = max(0, self.progress_idx - back)
        hi = min(n - 1, self.progress_idx + fwd)

        best_i = self.progress_idx
        best_score = 1e9
        best_d = 1e9

        HEADING_W = 0.35

        for i in range(lo, hi + 1):
            px, py = self.global_path[i]
            d = math.hypot(px - self.current_x, py - self.current_y)

            sh = self.seg_heading(i)
            hm = abs(wrap_pi(sh - self.current_yaw))

            score = d + HEADING_W * hm

            if score < best_score:
                best_score = score
                best_i = i
                best_d = d

        if best_i >= self.progress_idx:
            self.progress_idx = best_i
        else:
            self.progress_idx = max(0, self.progress_idx - 1)

        return best_i, best_d

    def find_lookahead_on_path(self, start_idx):
        n = len(self.global_path)
        if n == 0:
            return (self.current_x, self.current_y), 0
        if start_idx >= n - 1:
            return self.global_path[-1], n - 1

        accumulated = 0.0
        prev_x, prev_y = self.global_path[start_idx]

        for i in range(start_idx + 1, n):
            px, py = self.global_path[i]
            seg = math.hypot(px - prev_x, py - prev_y)
            accumulated += seg
            if accumulated >= self.lookahead_distance:
                return (px, py), i
            prev_x, prev_y = px, py

        return self.global_path[-1], n - 1

    # ============================================================
    # LiDAR helpers
    # ============================================================
    def angle_to_index(self, ang):
        idx = int((ang - self.ANG_MIN) / self.ANG_RES)
        return int(clamp(idx, 0, self.N_BINS - 1))

    def compute_sector_min(self, center_ang, half_width_deg):
        hw = math.radians(half_width_deg)
        a0 = center_ang - hw
        a1 = center_ang + hw
        i0 = self.angle_to_index(a0)
        i1 = self.angle_to_index(a1)
        if i0 > i1:
            i0, i1 = i1, i0
        return min(self.ranges[i0:i1 + 1]) if i1 >= i0 else self.MAX_RANGE

    def compute_front_min(self):
        return self.compute_sector_min(0.0, 30.0)

    def smooth_ranges(self, ranges, k=5):
        if k <= 1:
            return ranges[:]
        half = k // 2
        out = ranges[:]
        n = len(ranges)
        for i in range(n):
            s = 0.0
            c = 0
            for j in range(i - half, i + half + 1):
                if 0 <= j < n:
                    s += ranges[j]
                    c += 1
            out[i] = s / max(1, c)
        return out

    def find_largest_gaps(self, free_mask):
        gaps = []
        n = len(free_mask)
        i = 0
        while i < n:
            if free_mask[i]:
                s = i
                while i + 1 < n and free_mask[i + 1]:
                    i += 1
                e = i
                gaps.append((s, e))
            i += 1
        return gaps

    def pick_gap_angle(self, desired_angle, max_dev_deg=35.0):
        """
        goal-biased + path-faithful gap 선택:
        - desired_angle 근처 갭만 우선 고려 (너무 큰 우회 방지)
        - 그래도 없으면 그때만 전체에서 선택
        """
        ranges = [clamp(r, 0.0, self.MAX_RANGE) for r in self.ranges]
        ranges = self.smooth_ranges(ranges, k=5)

        free = [r > self.GAP_CLEAR_DIST for r in ranges]

        # bubble around closest obstacle near desired direction (+/- 50deg)
        search_hw = math.radians(50.0)
        i0 = self.angle_to_index(desired_angle - search_hw)
        i1 = self.angle_to_index(desired_angle + search_hw)
        if i0 > i1:
            i0, i1 = i1, i0

        closest_i = None
        closest_r = 1e9
        for i in range(i0, i1 + 1):
            if ranges[i] < closest_r:
                closest_r = ranges[i]
                closest_i = i

        if closest_i is not None and closest_r < 3.5:
            angle_span = math.atan2(self.BUBBLE_RADIUS, max(0.01, closest_r))
            bubble_bins = int(angle_span / max(1e-6, self.ANG_RES)) + 1
            for j in range(closest_i - bubble_bins, closest_i + bubble_bins + 1):
                if 0 <= j < self.N_BINS:
                    free[j] = False

        gaps = self.find_largest_gaps(free)
        if not gaps:
            return 0.0

        # 1) 우선: desired 근처 갭만 사용
        max_dev = math.radians(max_dev_deg)
        candidate = []
        for s, e in gaps:
            c = (s + e) // 2
            ang_c = self.ANG_MIN + c * self.ANG_RES
            if abs(wrap_pi(ang_c - desired_angle)) <= max_dev:
                candidate.append((s, e, ang_c))

        use = candidate if candidate else [(s, e, self.ANG_MIN + ((s + e)//2) * self.ANG_RES) for (s, e) in gaps]

        best_ang = 0.0
        best_score = -1e9

        for s, e, ang_c in use:
            width = e - s + 1
            align = abs(wrap_pi(ang_c - desired_angle))
            hyster = abs(wrap_pi(ang_c - self.prev_gap_angle))

            # align 페널티를 강하게 + 히스테리시스로 좌우 흔들림 감소
            score = width - 55.0 * align - 10.0 * hyster

            if score > best_score:
                best_score = score
                best_ang = ang_c

        best_ang = clamp(best_ang, -math.radians(70.0), math.radians(70.0))
        self.prev_gap_angle = best_ang
        return best_ang

    # ============================================================
    # Path command (Regulated + CTE)
    # ============================================================
    def compute_path_command(self):
        closest_idx, dist_to_path = self.find_closest_point_on_path_windowed()
        (tx, ty), _ = self.find_lookahead_on_path(closest_idx)

        dx = tx - self.current_x
        dy = ty - self.current_y
        dist_to_target = math.hypot(dx, dy)

        angle_to_target = math.atan2(dy, dx)
        heading_err = wrap_pi(angle_to_target - self.current_yaw)

        cte = self.signed_cross_track(closest_idx)

        k_h = 2.2
        k_cte = 2.2
        w = k_h * heading_err - k_cte * cte
        w = clamp(w, -self.max_w, self.max_w)

        v = min(self.max_v, 1.2 * dist_to_target)

        if abs(heading_err) > math.radians(75):
            v = 0.05
            w = clamp(2.4 * heading_err, -self.max_w, self.max_w)

        turn_slow = clamp(1.0 - 0.45 * (abs(w) / self.max_w), 0.35, 1.0)
        v *= turn_slow
        v = clamp(v, 0.05, self.max_v)

        return v, w, heading_err, dist_to_path, cte

    # ============================================================
    # Avoid command (Path-faithful: cap deviation + fast rejoin)
    # ============================================================
    def compute_avoid_command(self, desired_angle, dist_to_path, hazard_path, front_min, w_path, cte):
        # emergency: stop+turn
        if front_min < self.EMERGENCY_DIST:
            theta_gap = self.pick_gap_angle(desired_angle, max_dev_deg=80.0)
            w = clamp(2.2 * theta_gap, -self.max_w, self.max_w)
            v = 0.0
            self.mode = "EMERGENCY"
            return v, w

        theta_gap = self.pick_gap_angle(desired_angle, max_dev_deg=35.0)

        # (A) 회피각이 경로각에서 너무 멀어지지 않도록 캡
        max_delta = math.radians(self.MAX_AVOID_DELTA_DEG)
        delta = wrap_pi(theta_gap - desired_angle)
        delta = clamp(delta, -max_delta, max_delta)
        theta = wrap_pi(desired_angle + delta)

        # (B) hazard가 심할수록 delta를 더 쓰고, 조금 막히면 거의 경로대로
        # hazard_path가 AVOID_CLEAR_PATH에 가까우면 w_avoid_ratio 작아져서 바로 복귀 느낌
        w_avoid_ratio = clamp((self.AVOID_CLEAR_PATH - hazard_path) / max(1e-6, (self.AVOID_CLEAR_PATH - self.AVOID_START_PATH)), 0.0, 1.0)
        w_avoid_ratio = clamp(w_avoid_ratio, 0.10, 1.0)

        w_gap_cmd = clamp(2.2 * theta, -self.max_w, self.max_w)

        # (C) 최종 조향은 "경로 조향"을 베이스로, 필요한 만큼만 gap 방향으로 이동
        w = w_path + w_avoid_ratio * (w_gap_cmd - w_path)

        # (D) 경로 복귀를 더 빠르게: dist_to_path 커지면 cte 복귀 항 강화
        if dist_to_path > 0.4:
            w += (-1.3 * cte)

        w = clamp(w, -self.max_w, self.max_w)

        # speed: front_min이 가까울수록만 감속 (path_clear가 뚫리면 바로 속도 회복)
        v_max_avoid = min(0.52, self.max_v * 0.90)
        slow = clamp((front_min - 0.6) / 2.0, 0.35, 1.0)
        v = v_max_avoid * slow
        v *= clamp(1.0 - 0.55 * (abs(w) / self.max_w), 0.30, 1.0)
        v = clamp(v, 0.05, v_max_avoid)

        self.mode = "GAP_AVOID"
        return v, w

    # ============================================================
    # Stuck escape (reverse only here)
    # ============================================================
    def check_stuck_and_escape(self, front_min):
        now = self.get_clock().now()
        dt = (now - self.last_progress_check_time).nanoseconds / 1e9
        if dt < 2.0:
            return False

        moved = self.total_distance_traveled - self.last_progress_distance
        self.last_progress_check_time = now
        self.last_progress_distance = self.total_distance_traveled

        if moved < 0.04 and front_min < self.AVOID_START_PATH:
            left = sum(self.ranges[self.N_BINS//2:]) / (self.N_BINS//2)
            right = sum(self.ranges[:self.N_BINS//2]) / (self.N_BINS//2)
            self.escape_turn_dir = +1.0 if left > right else -1.0
            self.escape_until = now.nanoseconds / 1e9 + 1.0
            self.mode = "ESCAPE"
            return True

        return False

    # ============================================================
    # Control loop
    # ============================================================
    def control_loop(self):
        if not self.simulation_running or self.path_completed:
            return
        if not self.path_generated or len(self.global_path) == 0:
            return

        # ESCAPE override
        if self.escape_until is not None:
            tnow = self.get_clock().now().nanoseconds / 1e9
            if tnow < self.escape_until:
                cmd = Twist()
                cmd.linear.x = -0.12
                cmd.angular.z = float(self.escape_turn_dir * 1.4)
                self.apply_smoothing_and_publish(cmd)
                return
            else:
                self.escape_until = None
                self.mode = "GAP_AVOID"

        # update clearance
        self.front_min_dist = self.compute_front_min()

        # path tracking command
        v_path, w_path, heading_err, dist_to_path, cte = self.compute_path_command()
        desired_angle = clamp(heading_err, -math.radians(80), math.radians(80))

        # path 진행방향 기준으로 "진짜 막혔는지" 판단 (핵심)
        self.path_clear_dist = self.compute_sector_min(desired_angle, 12.0)
        hazard_path = self.path_clear_dist  # ✅ front에 끌려다니지 않게

        # stuck -> escape
        if self.check_stuck_and_escape(self.front_min_dist):
            return

        # mode switching (fast rejoin)
        if self.mode == "TRACK":
            if hazard_path < self.AVOID_START_PATH:
                self.mode = "GAP_AVOID"
        else:
            # path가 뚫리면 즉시 TRACK 복귀
            if hazard_path > self.AVOID_CLEAR_PATH:
                self.mode = "TRACK"

        # compute final cmd
        if self.mode == "TRACK":
            cmd = Twist()
            cmd.linear.x = float(v_path)
            cmd.angular.z = float(w_path)
        else:
            v_avoid, w_avoid = self.compute_avoid_command(
                desired_angle=desired_angle,
                dist_to_path=dist_to_path,
                hazard_path=hazard_path,
                front_min=self.front_min_dist,
                w_path=w_path,
                cte=cte
            )
            cmd = Twist()
            cmd.linear.x = float(v_avoid)
            cmd.angular.z = float(w_avoid)

            # 너무 멀리 drift 하면 복귀 성향 추가(안전)
            if dist_to_path > self.MAX_PATH_DEVIATION:
                cmd.angular.z = float(0.70 * cmd.angular.z + 0.30 * w_path)

        self.apply_smoothing_and_publish(cmd)

    def apply_smoothing_and_publish(self, cmd: Twist):
        self.cmd_filt_v = (1 - self.alpha_v) * self.cmd_filt_v + self.alpha_v * cmd.linear.x
        self.cmd_filt_w = (1 - self.alpha_w) * self.cmd_filt_w + self.alpha_w * cmd.angular.z

        cmd.linear.x = float(self.cmd_filt_v)
        cmd.angular.z = float(self.cmd_filt_w)

        self.cmd_vel_pub.publish(cmd)
        self.current_action = {'linear_x': cmd.linear.x, 'angular_z': cmd.angular.z}

    # ============================================================
    # LiDAR callback
    # ============================================================
    def lidar_callback(self, msg: PointCloud2):
        ranges = [self.MAX_RANGE] * self.N_BINS

        try:
            point_step = msg.point_step
            data = msg.data
            stride = point_step * 10

            for off in range(0, len(data), stride):
                if off + 12 > len(data):
                    break

                x = struct.unpack_from('f', data, off + 0)[0]
                y = struct.unpack_from('f', data, off + 4)[0]
                z = struct.unpack_from('f', data, off + 8)[0]

                if not (0.15 < x < self.MAX_RANGE and abs(y) < 3.0 and -0.45 < z < 0.15):
                    continue

                r = math.hypot(x, y)
                if r < 0.05:
                    continue

                ang = math.atan2(y, x)
                if ang < self.ANG_MIN or ang > self.ANG_MAX:
                    continue

                idx = int((ang - self.ANG_MIN) / self.ANG_RES)
                if 0 <= idx < self.N_BINS and r < ranges[idx]:
                    ranges[idx] = r

            self.ranges = ranges
            self.min_obstacle_distance = min(ranges)
            min_idx = ranges.index(self.min_obstacle_distance)
            self.obstacle_angle = self.ANG_MIN + min_idx * self.ANG_RES

        except Exception:
            self.ranges = [self.MAX_RANGE] * self.N_BINS
            self.min_obstacle_distance = 999.0
            self.obstacle_angle = 0.0

        self.current_sample['lidar'] = {
            'timestamp': msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9,
            'width': msg.width,
            'height': msg.height,
            'min_distance': float(self.min_obstacle_distance),
            'front_min': float(self.front_min_dist if self.front_min_dist < 900 else 999.0),
            'path_clear': float(self.path_clear_dist if self.path_clear_dist < 900 else 999.0),
            'mode': self.mode
        }
        self.try_save_sample()

    # ============================================================
    # Camera callback
    # ============================================================
    def camera_callback(self, msg: Image):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            img_filename = f"img_{self.sample_count:06d}.jpg"
            img_path = os.path.join(self.images_dir, img_filename)
            cv2.imwrite(img_path, cv_image)

            self.current_sample['camera'] = {
                'image_path': f"images/{img_filename}",
                'timestamp': msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9,
                'sensor_offset': {'x': 0.2, 'y': 0.0, 'z': 0.2}
            }
            self.try_save_sample()
        except Exception:
            pass

    # ============================================================
    # Odom callback
    # ============================================================
    def odom_callback(self, msg: Odometry):
        if not self.odom_received:
            self.odom_received = True
            self.get_logger().info("📡 Odom OK")

        self.last_odom_time = self.get_clock().now()

        self.current_x = float(msg.pose.pose.position.x)
        self.current_y = float(msg.pose.pose.position.y)
        self.current_z = float(msg.pose.pose.position.z)

        if self.start_x is None:
            self.start_x = self.current_x
            self.start_y = self.current_y
            self.last_x = self.current_x
            self.last_y = self.current_y
            self.get_logger().info(f"📍 Start: ({self.start_x:.2f}, {self.start_y:.2f})")

            self.global_path = self.generate_path_from_start(self.start_x, self.start_y)
            self.path_generated = True
            self.progress_idx = 0

            n = len(self.global_path)
            self.closest_search_window = max(18, int(0.12 * n))
            self.get_logger().info(f"🔧 closest_search_window={self.closest_search_window} (n={n})")

            self.publish_path()
            self.publish_path_markers()

            self.get_logger().info(f"📏 Z={self.current_z:.3f}")

        if self.last_x is not None:
            dx = self.current_x - self.last_x
            dy = self.current_y - self.last_y
            dist = math.hypot(dx, dy)
            self.total_distance_traveled += dist
            self.last_x = self.current_x
            self.last_y = self.current_y

        qx = msg.pose.pose.orientation.x
        qy = msg.pose.pose.orientation.y
        qz = msg.pose.pose.orientation.z
        qw = msg.pose.pose.orientation.w
        self.current_yaw = float(self.quaternion_to_yaw(qx, qy, qz, qw))

        self.current_sample['odometry'] = {
            'x': self.current_x,
            'y': self.current_y,
            'z': self.current_z,
            'qx': float(qx),
            'qy': float(qy),
            'qz': float(qz),
            'qw': float(qw),
            'yaw': float(self.current_yaw),
        }

        self.current_sample['action'] = self.current_action.copy()
        self.try_save_sample()

    # ============================================================
    # Save logic
    # ============================================================
    def try_save_sample(self):
        required = ['lidar', 'camera', 'action', 'odometry']
        if all(k in self.current_sample for k in required):
            self.data_buffer.append(self.current_sample.copy())
            self.sample_count += 1

            if self.sample_count % 100 == 0:
                self.get_logger().info(f"📊 {self.sample_count}")
                self.save_to_file()

            self.current_sample = {}

    def save_to_file(self):
        if not self.data_buffer:
            return

        filename = f"data_batch_{self.sample_count}.json"
        filepath = os.path.join(self.session_dir, filename)

        with open(filepath, 'w') as f:
            json.dump(self.data_buffer, f, indent=2)

        self.get_logger().info(f"💾 {filename}")
        self.data_buffer = []


def main(args=None):
    rclpy.init(args=args)
    collector = Nav2DataCollector()

    try:
        rclpy.spin(collector)
    except (KeyboardInterrupt, SystemExit):
        try:
            collector.save_to_file()
        except Exception:
            pass
        try:
            collector.cmd_vel_pub.publish(Twist())
        except Exception:
            pass
    finally:
        try:
            collector.destroy_node()
        except Exception:
            pass
        try:
            if rclpy.ok():
                rclpy.shutdown()
        except Exception:
            pass


if __name__ == '__main__':
    main()
