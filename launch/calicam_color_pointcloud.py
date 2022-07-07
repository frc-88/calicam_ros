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
                    plugin='depth_image_proc::PointCloudXyzrgbNode',
                    name='point_cloud_xyzrgb_node',
                    remappings=[('rgb/camera_info', '/calicam/left/camera_info'),
                                ('rgb/image_rect_color', '/calicam/left/color_rect'),
                                ('depth_registered/image_rect', '/calicam/depth/image_rect'),
                                ('points', '/calicam/depth/points')]
                ),
            ],
            output='screen',
        )
    ])