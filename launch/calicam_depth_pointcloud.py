from launch import LaunchDescription

import launch_ros.actions
import launch_ros.descriptions


def generate_launch_description():
    return LaunchDescription([
        # launch plugin through rclcpp_components container
        launch_ros.actions.ComposableNodeContainer(
            name='calicam_container',
            namespace='',
            package='rclcpp_components',
            executable='component_container',
            composable_node_descriptions=[
                # Driver itself
                launch_ros.descriptions.ComposableNode(
                    package='depth_image_proc',
                    plugin='depth_image_proc::PointCloudXyzNode',
                    name='point_cloud_xyz_node',
                    remappings=[('image_rect', '/calicam/depth/image_rect'),
                                ('camera_info', '/calicam/depth/camera_info'),
                                ('points', '/calicam/depth/points')]
                ),
            ],
            output='screen',
        )
    ])