#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from std_msgs.msg import Float64MultiArray
import numpy as np

class ScoutMiniController(Node):
    def __init__(self):
        super().__init__('scout_mini_controller')
        
        # Scout Mini 파라미터
        self.wheel_track = 0.498   # 좌우 바퀴 간격 (m)
        self.wheel_radius = 0.165  # 바퀴 반지름 (m)
        
        # Subscriber: cmd_vel 받기
        self.cmd_vel_sub = self.create_subscription(
            Twist, 
            '/cmd_vel', 
            self.cmd_vel_callback, 
            10
        )
        
        # Publisher: Isaac Sim 바퀴에 직접 명령
        self.wheel_pub = self.create_publisher(
            Float64MultiArray, 
            '/scout_mini/wheel_commands', 
            10
        )
        
        self.get_logger().info("Scout Mini Controller started (4WD Skid-Steer mode)")
        self.get_logger().info(f"Subscribed to: /cmd_vel")
        self.get_logger().info(f"Publishing to: /scout_mini/wheel_commands")
        self.get_logger().info(f"Wheel track: {self.wheel_track}m, Wheel radius: {self.wheel_radius}m")
    
    def cmd_vel_callback(self, msg):
        """
        cmd_vel 메시지를 받아서 4개 바퀴 속도 계산
        Skid-Steer 방식: 양쪽 바퀴가 반대 방향으로 회전
        """
        linear_x = msg.linear.x
        angular_z = msg.angular.z
        
        # Skid-Steer Differential Drive
        # 왼쪽 바퀴: 전진 - 회전
        # 오른쪽 바퀴: 전진 + 회전
        left_velocity = linear_x - (angular_z * self.wheel_track / 2.0)
        right_velocity = linear_x + (angular_z * self.wheel_track / 2.0)
        
        # 선속도 -> 각속도 변환 (rad/s)
        left_angular_vel = left_velocity / self.wheel_radius
        right_angular_vel = right_velocity / self.wheel_radius
        
        # ⭐ 왼쪽 바퀴 부호 반전 (URDF 특성)
        left_angular_vel = -left_angular_vel
        
        # 4개 바퀴 명령 (Skid-Steer: 좌우 동일)
        wheel_cmd = Float64MultiArray()
        wheel_cmd.data = [
            left_angular_vel,   # front left
            right_angular_vel,  # front right
            left_angular_vel,   # rear left
            right_angular_vel   # rear right
        ]
        
        self.wheel_pub.publish(wheel_cmd)
        
        # 디버그 출력
        if abs(linear_x) > 0.01 or abs(angular_z) > 0.01:
            # 제자리 회전 체크
            if abs(linear_x) < 0.01 and abs(angular_z) > 0.01:
                self.get_logger().info(
                    f"🔄 ROTATE: angular={angular_z:.2f} | "
                    f"LEFT={left_angular_vel:.2f} (←), RIGHT={right_angular_vel:.2f} (→)"
                )
            else:
                self.get_logger().info(
                    f"cmd_vel: linear={linear_x:.2f}, angular={angular_z:.2f} | "
                    f"wheels: L={left_angular_vel:.2f}, R={right_angular_vel:.2f}"
                )

def main(args=None):
    rclpy.init(args=args)
    controller = ScoutMiniController()
    
    try:
        rclpy.spin(controller)
    except KeyboardInterrupt:
        pass
    finally:
        controller.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()