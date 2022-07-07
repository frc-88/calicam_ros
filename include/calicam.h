#include <iostream>
#include <iomanip>
#include <string>
#include <chrono>

#ifdef CUDA_STEREO_SGM
#include <cuda_runtime.h>
#include <libsgm.h>
#endif
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/core/version.hpp>



struct device_buffer
{
    device_buffer() : data(nullptr) {}
    device_buffer(size_t count) {
#ifdef CUDA_STEREO_SGM
        allocate(count);
#endif
    }
    void allocate(size_t count) {
#ifdef CUDA_STEREO_SGM
        cudaMalloc(&data, count);
#endif
    }
    ~device_buffer() {
#ifdef CUDA_STEREO_SGM
        cudaFree(data);
#endif
    }
    void* data;
};

template <class... Args>
static std::string format_string(const char* fmt, Args... args);

class StereoCalicam
{
private:
    cv::VideoCapture capture;
    int capture_num;

    double fps;
    std::string cam_model;
    
    std::string param_path;
    
    int capture_width, capture_height;
    int image_width, image_height, image_channels;

    double max_depth;

    cv::Mat raw, raw_left, raw_right, rect_left, rect_right;
    cv::Mat gray_left, gray_right;
    cv::Mat disparity, depth;

    int num_disparities;

    bool calculate_stereo;
    bool enable_undistort;
    bool use_cuda;

    double scale_ratio_w, scale_ratio_h;

    double virtual_fov;
    cv::Mat Translation, Kl, Kr, Dl, Dr, xil, xir, Rl, Rr, smap[2][2], Knew;

    int invalid_disp;
#ifdef CUDA_STEREO_SGM
    sgm::StereoSGM* cuda_sgm;
	device_buffer* d_I1;
    device_buffer* d_I2;
    device_buffer* d_disparity;
    int input_bytes, output_bytes;
#endif

    cv::Ptr<cv::StereoSGBM> cpu_sgbm;
    int disparity_window;
    int pre_filter_cap;
    int min_disparity;
    int texture_threshold;
    int uniqueness_ratio;
    int speckle_window_size;
    int speckle_range;
    int disp12_max_diff;

    double pipeline_fps;

    void init_undistort_rectify_map(cv::Mat K, cv::Mat D, cv::Mat xi, cv::Mat R,
                                    cv::Mat P, cv::Size size,
                                    cv::Mat &map1, cv::Mat &map2);
    void init_reproject_mat(cv::Mat K, double baseline);

    void init_rectify_map();
    void load_parameters(std::string file_name);

    void compute_depth();

    std::vector<double> mat_to_vector(cv::Mat mat);

public:
    const double cal_width = 1280.0;
    const double cal_height = 960.0;

    StereoCalicam(int capture_num, std::string param_path);
    void begin();
    int get_width()  { return image_width; }
    int get_height()  { return image_height; }
    double get_pipeline_fps()  { return pipeline_fps; }
    double get_fps()  { return fps; }
    double get_max_depth()  { return max_depth; }

    cv::Mat get_raw()  { return raw; }
    cv::Mat get_left_raw()  { return raw_left; }
    cv::Mat get_right_raw()  { return raw_right; }
    cv::Mat get_left_rect()  { return rect_left; }
    cv::Mat get_right_rect()  { return rect_right; }
    cv::Mat get_disparity()  { return disparity; }
    cv::Mat get_depth()  { return depth; }
    cv::Mat get_depth_mm() {
        cv::Mat depth_mm;
        depth.convertTo(depth_mm, CV_16UC1, 1000.0);
        return depth_mm;
    }

    cv::Mat get_debug_disparity();
    double get_distance(int x, int y, int radius);
    cv::Mat get_debug_image();

    std::vector<double> get_left_distortion();
    std::vector<double> get_left_camera_matrix();
    std::vector<double> get_left_rectification();
    std::vector<double> get_left_projection();
    std::vector<double> get_right_distortion();
    std::vector<double> get_right_camera_matrix();
    std::vector<double> get_right_rectification();
    std::vector<double> get_right_projection();
    std::vector<double> get_translation_left_to_right();

    void process();

    ~StereoCalicam();
};
