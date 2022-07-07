#include "calicam.h"

#include "rclcpp/rclcpp.hpp"

#include <boost/range/algorithm.hpp>

#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/camera_info.hpp>

#include <cv_bridge/cv_bridge.h>

#include <image_transport/image_transport.hpp>


#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include <tf2_ros/transform_broadcaster.h>
#include <tf2_ros/buffer.h>
#include <tf2/exceptions.h>

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

    node->declare_parameter<bool>("debug_show", 0);
    bool debug_show = node->get_parameter("debug_show").as_bool();

    node->declare_parameter<bool>("publish_tf", true);
    bool publish_tf = node->get_parameter("publish_tf").as_bool();

    StereoCalicam cam(capture_num, param_path);
    cam.begin();

    rmw_qos_profile_t custom_qos_profile = rmw_qos_profile_sensor_data;
    custom_qos_profile.depth = 1;

    image_transport::ImageTransport image_transport(node);
    image_transport::Publisher left_raw_camera = image_transport::create_publisher(node.get(), "calicam/left/color_raw", custom_qos_profile);
    image_transport::Publisher right_raw_camera = image_transport::create_publisher(node.get(), "calicam/right/color_raw", custom_qos_profile);

    image_transport::CameraPublisher left_rect_camera = image_transport::create_camera_publisher(node.get(), "calicam/left/color_rect", custom_qos_profile);
    image_transport::CameraPublisher right_rect_camera = image_transport::create_camera_publisher(node.get(), "calicam/right/color_rect", custom_qos_profile);

    image_transport::CameraPublisher depth_camera = image_transport::create_camera_publisher(node.get(), "calicam/depth/image_rect", custom_qos_profile);

    std_msgs::msg::Header left_header;
    std_msgs::msg::Header right_header;
    std_msgs::msg::Header depth_header;

    left_header.frame_id = camera_prefix + "_left";
    right_header.frame_id = camera_prefix + "_right";
    depth_header.frame_id = camera_prefix + "_left";

    std::string base_frame_id = camera_prefix + "_base";
    std::string world_frame_id = camera_prefix + "_world";

    std::shared_ptr<sensor_msgs::msg::CameraInfo> base_info(new sensor_msgs::msg::CameraInfo());
    std::shared_ptr<sensor_msgs::msg::CameraInfo> left_rect_info(new sensor_msgs::msg::CameraInfo());
    std::shared_ptr<sensor_msgs::msg::CameraInfo> right_rect_info(new sensor_msgs::msg::CameraInfo());
    std::shared_ptr<sensor_msgs::msg::CameraInfo> depth_rect_info(new sensor_msgs::msg::CameraInfo());

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

    left_rect_info->header = left_header;
    right_rect_info->header = right_header;
    depth_rect_info->header = depth_header;

    rclcpp::Rate loop_rate(cam.get_fps());

    std::unique_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster;
    geometry_msgs::msg::TransformStamped left_stamped, right_stamped, depth_stamped, world_stamped;
    if (publish_tf) {
        tf_broadcaster = std::make_unique<tf2_ros::TransformBroadcaster>(*node);

        tf2::Quaternion base_quat;
        base_quat.setRPY(0.0, 0.0, 0.0);

        tf2::Quaternion world_quat;
        world_quat.setRPY(-M_PI/2.0, 0.0, -M_PI/2.0);

        geometry_msgs::msg::Quaternion msg_base_quat = tf2::toMsg(base_quat);
        geometry_msgs::msg::Quaternion msg_world_quat = tf2::toMsg(world_quat);

        std::vector<double> translation = cam.get_translation_left_to_right();

        left_stamped.header.frame_id = base_frame_id;
        left_stamped.child_frame_id = left_header.frame_id;
        left_stamped.transform.translation.x = translation.at(0) / 2.0;
        left_stamped.transform.translation.y = translation.at(1) / 2.0;
        left_stamped.transform.translation.z = translation.at(2) / 2.0;
        left_stamped.transform.rotation = msg_base_quat;

        right_stamped.header.frame_id = base_frame_id;
        right_stamped.child_frame_id = right_header.frame_id;
        right_stamped.transform.translation.x = -translation.at(0) / 2.0;
        right_stamped.transform.translation.y = -translation.at(1) / 2.0;
        right_stamped.transform.translation.z = -translation.at(2) / 2.0;
        right_stamped.transform.rotation = msg_base_quat;

        depth_stamped = left_stamped;
        depth_stamped.child_frame_id = depth_header.frame_id;

        world_stamped.header.frame_id = world_frame_id;
        world_stamped.child_frame_id = base_frame_id;
        world_stamped.transform.translation.x = 0.0;
        world_stamped.transform.translation.y = 0.0;
        world_stamped.transform.translation.z = 0.0;
        world_stamped.transform.rotation = msg_world_quat;

    }

    std::string param_win_name("calicam");
    cv::Mat debug_image;
    int debug_index = 0;
    if (debug_show) {
        cv::namedWindow(param_win_name);
    }

    while (rclcpp::ok())
    {
        cam.process();

        rclcpp::Time now = node->get_clock()->now();
        left_header.stamp = now;
        right_header.stamp = now;
        depth_header.stamp = now;

        std::shared_ptr<sensor_msgs::msg::Image> left_raw_msg = cv_bridge::CvImage(left_header, sensor_msgs::image_encodings::BGR8, cam.get_left_raw()).toImageMsg();
        std::shared_ptr<sensor_msgs::msg::Image> right_raw_msg = cv_bridge::CvImage(right_header, sensor_msgs::image_encodings::BGR8, cam.get_right_raw()).toImageMsg();

        std::shared_ptr<sensor_msgs::msg::Image> left_rect_msg = cv_bridge::CvImage(left_header, sensor_msgs::image_encodings::BGR8, cam.get_left_rect()).toImageMsg();
        std::shared_ptr<sensor_msgs::msg::Image> right_rect_msg = cv_bridge::CvImage(right_header, sensor_msgs::image_encodings::BGR8, cam.get_right_rect()).toImageMsg();

        std::shared_ptr<sensor_msgs::msg::Image> depth_msg = cv_bridge::CvImage(depth_header, sensor_msgs::image_encodings::TYPE_32FC1, cam.get_depth()).toImageMsg();
        // std::shared_ptr<sensor_msgs::msg::Image> depth_msg = cv_bridge::CvImage(depth_header, sensor_msgs::image_encodings::MONO16, cam.get_depth_mm()).toImageMsg();

        left_rect_info->header.stamp = now;
        right_rect_info->header.stamp = now;
        depth_rect_info->header.stamp = now;

        left_raw_camera.publish(left_raw_msg);
        right_raw_camera.publish(right_raw_msg);

        left_rect_camera.publish(left_rect_msg, left_rect_info);
        right_rect_camera.publish(right_rect_msg, right_rect_info);

        depth_camera.publish(depth_msg, depth_rect_info);

        if (publish_tf)
        {
            left_stamped.header.stamp = now;
            depth_stamped.header.stamp = now;
            right_stamped.header.stamp = now;
            world_stamped.header.stamp = now;

            tf_broadcaster->sendTransform(left_stamped);
            tf_broadcaster->sendTransform(depth_stamped);
            tf_broadcaster->sendTransform(right_stamped);
            tf_broadcaster->sendTransform(world_stamped);
        }

        rclcpp::spin_some(node);
        loop_rate.sleep();

        if (debug_show) {
            switch (debug_index)
            {
                case 0: debug_image = cam.get_left_rect(); break;
                case 1: debug_image = cam.get_right_rect(); break;
                case 2: debug_image = cam.get_left_raw(); break;
                case 3: debug_image = cam.get_right_raw(); break;
                case 4: debug_image = cam.get_debug_disparity(); break;
                case 5: cam.get_depth().convertTo(debug_image, CV_8UC1, 255.0 / cam.get_max_depth()); break;
                
                default:
                    break;
            }
            imshow(param_win_name, debug_image);
            char key = cv::waitKey(1);
            if (key == 'q' || key == 'Q' || key == 27) {
                break;
            }
            else {
                switch (key)
                {
                    case '1': debug_index = 0; break;
                    case '2': debug_index = 1; break;
                    case '3': debug_index = 2; break;
                    case '4': debug_index = 3; break;
                    case '5': debug_index = 4; break;
                    case '6': debug_index = 5; break;
                    default:
                        break;
                }
            }
        }

    }
    return 0;
}
