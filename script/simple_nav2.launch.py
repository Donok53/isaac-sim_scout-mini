#!/usr/bin/env python3

import os
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    
    params_file = os.path.join(os.path.dirname(__file__), 'isaac_nav2_params.yaml')
    
    return LaunchDescription([
        
        Node(
            package='nav2_controller',
            executable='controller_server',
            name='controller_server',
            output='screen',
            parameters=[params_file]
        ),
        
        Node(
            package='nav2_planner',
            executable='planner_server',
            name='planner_server',
            output='screen',
            parameters=[params_file]
        ),
        
        Node(
            package='nav2_behaviors',
            executable='behavior_server',
            name='behavior_server',
            output='screen',
            parameters=[params_file]
        ),
        
        Node(
            package='nav2_bt_navigator',
            executable='bt_navigator',
            name='bt_navigator',
            output='screen',
            parameters=[params_file]
        ),
        
        Node(
            package='nav2_lifecycle_manager',
            executable='lifecycle_manager',
            name='lifecycle_manager',
            output='screen',
            parameters=[{
                'use_sim_time': True,
                'autostart': True,
                'node_names': ['controller_server', 'planner_server', 'behavior_server', 'bt_navigator']
            }]
        ),
    ])