#include "calicam.h"

#include "rclcpp/rclcpp.hpp"

#include <boost/range/algorithm.hpp>

#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/camera_info.hpp>

#include <cv_bridge/cv_bridge.h>

#include <image_transport/image_transport.hpp>


template<size_t Size, class Container>
std::array<typename Container::value_type, Size> as_array(const Container &cont)
{
    assert(cont.size() == Size);
    std::array<typename Container::value_type, Size> result;
    boost::range::copy(cont, result.begin());
    return result;
}


int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    auto node = rclcpp::Node::make_shared("calicam");

    node->declare_parameter<int>("capture_num", 0);
    int capture_num = node->get_parameter("capture_num").as_int();

    node->declare_parameter<std::string>("param_path", "./astar_calicam.yml");
    std::string param_path = node->get_parameter("param_path").as_string();

    node->declare_parameter<std::string>("camera_prefix", "calicam");
    std::string camera_prefix = node->get_parameter("camera_prefix").as_string();

    StereoCalicam cam(capture_num, param_path);
    cam.begin();

    rmw_qos_profile_t custom_qos_profile = rmw_qos_profile_sensor_data;
    custom_qos_profile.depth = 1;

    image_transport::ImageTransport image_transport(node);
    image_transport::Publisher left_raw_camera = image_transport::create_publisher(node.get(), "calicam/left/color_raw", custom_qos_profile);
    image_transport::Publisher right_raw_camera = image_transport::create_publisher(node.get(), "calicam/right/color_raw", custom_qos_profile);

    image_transport::CameraPublisher left_rect_camera = image_transport::create_camera_publisher(node.get(), "calicam/left/color_rect", custom_qos_profile);
    image_transport::CameraPublisher right_rect_camera = image_transport::create_camera_publisher(node.get(), "calicam/right/color_rect", custom_qos_profile);

    image_transport::CameraPublisher depth_camera = image_transport::create_camera_publisher(node.get(), "calicam/depth_rect", custom_qos_profile);

    std_msgs::msg::Header left_header;
    std_msgs::msg::Header right_header;
    std_msgs::msg::Header depth_header;

    left_header.frame_id = camera_prefix + "_left";
    right_header.frame_id = camera_prefix + "_right";
    depth_header.frame_id = camera_prefix + "_depth";

    sensor_msgs::msg::CameraInfo::SharedPtr base_info;
    sensor_msgs::msg::CameraInfo::SharedPtr left_rect_info;
    sensor_msgs::msg::CameraInfo::SharedPtr right_rect_info;
    sensor_msgs::msg::CameraInfo::SharedPtr depth_rect_info;

    base_info->width = cam.get_width();
    base_info->height = cam.get_height();
    base_info->distortion_model = "fisheye";

    *left_rect_info = *base_info;
    *right_rect_info = *base_info;
    *depth_rect_info = *base_info;

    left_rect_info->d = cam.get_left_distortion();
    left_rect_info->k = as_array<9>(cam.get_left_camera_matrix());
    left_rect_info->r = as_array<9>(cam.get_left_rectification());
    left_rect_info->p = as_array<12>(cam.get_left_projection());
    right_rect_info->d = cam.get_right_distortion();
    right_rect_info->k = as_array<9>(cam.get_right_camera_matrix());
    right_rect_info->r = as_array<9>(cam.get_right_rectification());
    right_rect_info->p = as_array<12>(cam.get_right_projection());

    *depth_rect_info = *left_rect_info;

    rclcpp::Rate loop_rate(cam.get_fps());

    while (rclcpp::ok())
    {
        cam.process();

        rclcpp::Time now = node->get_clock()->now();
        left_header.stamp = now;
        right_header.stamp = now;
        depth_header.stamp = now;

        sensor_msgs::msg::Image::SharedPtr left_raw_msg = cv_bridge::CvImage(left_header, sensor_msgs::image_encodings::BGR8, cam.get_left_raw()).toImageMsg();
        sensor_msgs::msg::Image::SharedPtr right_raw_msg = cv_bridge::CvImage(right_header, sensor_msgs::image_encodings::BGR8, cam.get_right_raw()).toImageMsg();

        sensor_msgs::msg::Image::SharedPtr left_rect_msg = cv_bridge::CvImage(left_header, sensor_msgs::image_encodings::BGR8, cam.get_left_rect()).toImageMsg();
        sensor_msgs::msg::Image::SharedPtr right_rect_msg = cv_bridge::CvImage(right_header, sensor_msgs::image_encodings::BGR8, cam.get_right_rect()).toImageMsg();

        sensor_msgs::msg::Image::SharedPtr depth_msg = cv_bridge::CvImage(depth_header, sensor_msgs::image_encodings::MONO16, cam.get_depth_mm()).toImageMsg();

        left_raw_camera.publish(left_raw_msg);
        right_raw_camera.publish(right_raw_msg);

        left_rect_camera.publish(left_rect_msg, left_rect_info);
        right_rect_camera.publish(right_rect_msg, right_rect_info);

        depth_camera.publish(depth_msg, depth_rect_info);

        rclcpp::spin_some(node);
        loop_rate.sleep();
    }
    return 0;
}
