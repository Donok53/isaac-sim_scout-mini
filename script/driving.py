import time, threading
import numpy as np

# ------------------------
# 사용자 설정
# ------------------------
ROBOT_PRIM_CANDIDATES = ["/World/scout_mini", "/scout_mini"]
LEFT_WHEELS  = ["front_left_wheel", "rear_left_wheel"]
RIGHT_WHEELS = ["front_right_wheel", "rear_right_wheel"]

WHEEL_RADIUS = 0.10      # m  (대충값임. 너 로봇 값으로 수정 추천)
TRACK_WIDTH  = 0.40      # m  (좌우 바퀴 중심 간 거리. 대충값임)
CMD_TIMEOUT  = 0.3       # s  (이 시간 넘으면 정지)

# 네가 예전에 말한 규칙: "왼쪽: 음수(-)=앞, 오른쪽: 양수(+)=앞"
# 이게 맞다면 아래처럼 두면 됨. (틀리면 +/- 바꿔)
LEFT_SIGN  = -1.0
RIGHT_SIGN = +1.0

# ------------------------
# Isaac에서 Articulation 잡기
# ------------------------
from omni.isaac.core.utils.prims import is_prim_path_valid
from omni.isaac.core.articulations import Articulation

robot_prim = None
for p in ROBOT_PRIM_CANDIDATES:
    if is_prim_path_valid(p):
        robot_prim = p
        break

if robot_prim is None:
    raise RuntimeError(f"Robot prim not found. tried: {ROBOT_PRIM_CANDIDATES}")

art = Articulation(prim_path=robot_prim, name="scout")
art.initialize()

# DOF 인덱스 맵
dof_names = list(art.dof_names)
def idx(name: str) -> int:
    if name not in dof_names:
        raise RuntimeError(f"DOF '{name}' not found. available: {dof_names}")
    return dof_names.index(name)

left_idx  = [idx(n) for n in LEFT_WHEELS]
right_idx = [idx(n) for n in RIGHT_WHEELS]

# ------------------------
# ROS2 subscriber thread
# ------------------------
latest = {"v": 0.0, "w": 0.0, "t": time.time()}
lock = threading.Lock()

def ros_thread():
    import rclpy
    from rclpy.node import Node
    from geometry_msgs.msg import Twist

    rclpy.init(args=None)

    class CmdVelNode(Node):
        def __init__(self):
            super().__init__("isaac_cmdvel_bridge")
            self.create_subscription(Twist, "/cmd_vel", self.cb, 10)

        def cb(self, msg: Twist):
            with lock:
                latest["v"] = float(msg.linear.x)
                latest["w"] = float(msg.angular.z)
                latest["t"] = time.time()

    node = CmdVelNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

th = threading.Thread(target=ros_thread, daemon=True)
th.start()

# ------------------------
# 매 프레임 wheel velocity 적용 (Isaac 메인 스레드)
# ------------------------
import omni.kit.app
import omni.timeline

timeline = omni.timeline.get_timeline_interface()
app = omni.kit.app.get_app()

def apply_cmd():
    # 시뮬이 재생 중일 때만 적용(원하면 이 체크 제거)
    if not timeline.is_playing():
        return

    now = time.time()
    with lock:
        v, w, t = latest["v"], latest["w"], latest["t"]

    if (now - t) > CMD_TIMEOUT:
        v, w = 0.0, 0.0

    # diff drive: wheel angular velocity (rad/s)
    # v_l = (v - w*b/2) / r
    # v_r = (v + w*b/2) / r
    wl = (v - w * TRACK_WIDTH * 0.5) / max(WHEEL_RADIUS, 1e-6)
    wr = (v + w * TRACK_WIDTH * 0.5) / max(WHEEL_RADIUS, 1e-6)

    wl *= LEFT_SIGN
    wr *= RIGHT_SIGN

    # 4개 바퀴에 동일하게 넣기
    cmd = np.zeros(len(dof_names), dtype=np.float32)
    for i in left_idx:  cmd[i]  = wl
    for i in right_idx: cmd[i] = wr

    art.set_joint_velocities(cmd)

sub = app.get_update_event_stream().create_subscription_to_pop(lambda e: apply_cmd())

print("[OK] /cmd_vel subscriber running. Publish to /cmd_vel to drive the robot.")
print(f"robot_prim = {robot_prim}")
print(f"dof_names = {dof_names}")
