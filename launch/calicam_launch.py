import os
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    default_params = os.path.join(get_package_share_directory("calicam_ros"), "config", "21-181220-0172.yml")
    # default_params = get_package_share_directory() / "config" / "astar_calicam.yml"

    camera_config_path_arg = DeclareLaunchArgument(
        name='camera_config_path',
        default_value=default_params
    )
    camera_capture_num_arg = DeclareLaunchArgument(
        name='camera_capture_num',
        default_value="0"
    )
    camera_prefix_arg = DeclareLaunchArgument(
        name='camera_prefix',
        default_value="calicam"
    )

    return LaunchDescription([
        camera_config_path_arg,
        camera_capture_num_arg,
        camera_prefix_arg,
        Node(
            package="calicam_ros",
            # namespace="",
            executable="calicam_ros",
            name="calicam_ros",
            parameters=[
                {"param_path": LaunchConfiguration('camera_config_path')},
                {"capture_num": LaunchConfiguration('camera_capture_num')},
                {"camera_prefix": LaunchConfiguration('camera_prefix')},
            ]
        )
    ])
