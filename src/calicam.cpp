#include "calicam.h"

template <class... Args>
static std::string format_string(const char* fmt, Args... args)
{
    const int BUF_SIZE = 1024;
    char buf[BUF_SIZE];
    std::snprintf(buf, BUF_SIZE, fmt, args...);
    return std::string(buf);
}

inline double MatRowMul(cv::Mat m, double x, double y, double z, int r)
{
    return m.at<double>(r, 0) * x + m.at<double>(r, 1) * y + m.at<double>(r, 2) * z;
}

StereoCalicam::StereoCalicam(int capture_num, std::string param_path)
{
    image_channels = 3;
    this->capture_num = capture_num;
    this->param_path = param_path;
}

void StereoCalicam::begin()
{
    load_parameters(param_path);
    init_rectify_map();

    capture.open(capture_num);
    if (!capture.isOpened())
    {
        std::cout << "Camera doesn't work" << std::endl;
        exit(-1);
    }

    std::cout << "Setting capture width to " << capture_width << std::endl;
    std::cout << "Setting capture height to " << capture_height << std::endl;
    std::cout << "Setting capture fps to " << this->fps << std::endl;
    capture.set(cv::CAP_PROP_FRAME_WIDTH, capture_width);
    capture.set(cv::CAP_PROP_FRAME_HEIGHT, capture_height);
    capture.set(cv::CAP_PROP_FPS, this->fps);

    if (calculate_stereo)
    {
#ifdef CUDA_STEREO_SGM
        if (use_cuda)
        {
            int input_depth = 8;
            int output_depth = num_disparities < 256 ? 8 : 16;
            cuda_sgm = new sgm::StereoSGM(image_width, image_height, num_disparities, input_depth, output_depth, sgm::EXECUTE_INOUT_CUDA2CUDA);
            input_bytes = input_depth * image_width * image_height / 8;
            output_bytes = output_depth * image_width * image_height / 8;
            disparity = cv::Mat(cv::Size(image_width, image_height), output_depth == 8 ? CV_8U : CV_16U);

            invalid_disp = output_depth == 8
                ? static_cast< uint8_t>(cuda_sgm->get_invalid_disparity())
                : static_cast<uint16_t>(cuda_sgm->get_invalid_disparity());
            d_I1 = new device_buffer(input_bytes);
            d_I2 = new device_buffer(input_bytes);
            d_disparity = new device_buffer(output_bytes);
        }
        else
#endif  // CUDA_STEREO_SGM
        {
            invalid_disp = min_disparity - 1;
            disparity = cv::Mat(cv::Size(image_width, image_height), CV_16U);
            int N = num_disparities;
            int W = disparity_window;
            int C = image_channels;

            cpu_sgbm = cv::StereoSGBM::create(0, N / 8, W, 8 * C * W * W, 32 * C * W * W);
            cpu_sgbm->setPreFilterCap(pre_filter_cap);
            cpu_sgbm->setMinDisparity(min_disparity);
            cpu_sgbm->setUniquenessRatio(uniqueness_ratio);
            cpu_sgbm->setSpeckleWindowSize(speckle_window_size);
            cpu_sgbm->setSpeckleRange(speckle_range);
            cpu_sgbm->setDisp12MaxDiff(disp12_max_diff);
        }
        depth = cv::Mat(cv::Size(image_width, image_height), CV_32FC1);
    }
}

std::vector<double> StereoCalicam::mat_to_vector(cv::Mat mat)
{
    std::vector<double> vec;
    for (int r = 0; r < mat.rows; r++) {
        for (int c = 0; c < mat.cols; c++) {
            vec.push_back(mat.at<double>(r, c));
        }
    }
    return vec;
}

std::vector<double> StereoCalicam::get_left_distortion() {
    return mat_to_vector(Dl);
}

std::vector<double> StereoCalicam::get_left_camera_matrix() {
    return mat_to_vector(Knew);
}

std::vector<double> StereoCalicam::get_left_rectification() {
    return mat_to_vector(Rl);
}

std::vector<double> StereoCalicam::get_left_projection()
{
    std::vector<double> vec;
    vec.push_back(Knew.at<double>(0, 0));
    vec.push_back(Knew.at<double>(0, 1));
    vec.push_back(Knew.at<double>(0, 2));
    vec.push_back(Translation.at<double>(0, 0));
    vec.push_back(Knew.at<double>(1, 0));
    vec.push_back(Knew.at<double>(1, 1));
    vec.push_back(Knew.at<double>(1, 2));
    vec.push_back(Translation.at<double>(1, 0));
    vec.push_back(Knew.at<double>(2, 0));
    vec.push_back(Knew.at<double>(2, 1));
    vec.push_back(Knew.at<double>(2, 2));
    vec.push_back(Translation.at<double>(2, 0));
    return vec;
}


std::vector<double> StereoCalicam::get_right_distortion() {
    return mat_to_vector(Dr);
}

std::vector<double> StereoCalicam::get_right_camera_matrix(){
    return mat_to_vector(Knew);
}

std::vector<double> StereoCalicam::get_right_rectification() {
    return mat_to_vector(Rr);
}

std::vector<double> StereoCalicam::get_right_projection()
{
    std::vector<double> vec;
    vec.push_back(Knew.at<double>(0, 0));
    vec.push_back(Knew.at<double>(0, 1));
    vec.push_back(Knew.at<double>(0, 2));
    vec.push_back(0.0);
    vec.push_back(Knew.at<double>(1, 0));
    vec.push_back(Knew.at<double>(1, 1));
    vec.push_back(Knew.at<double>(1, 2));
    vec.push_back(0.0);
    vec.push_back(Knew.at<double>(2, 0));
    vec.push_back(Knew.at<double>(2, 1));
    vec.push_back(Knew.at<double>(2, 2));
    vec.push_back(0.0);
    return vec;
}

std::vector<double> StereoCalicam::get_translation_left_to_right()
{
    std::vector<double> vec;
    vec.push_back(Translation.at<double>(0, 0));
    vec.push_back(Translation.at<double>(1, 0));
    vec.push_back(Translation.at<double>(2, 0));
    return vec;
}

void StereoCalicam::load_parameters(std::string file_name)
{
    cv::FileStorage fs(file_name, cv::FileStorage::READ);
    if (!fs.isOpened())
    {
        std::cout << "Failed to open ini parameters: " << file_name << std::endl;
        exit(-1);
    }

    cv::Size cap_size;
    fs["cam_model"] >> cam_model;
    fs["cap_size"] >> cap_size;
    fs["Kl"] >> Kl;
    fs["Dl"] >> Dl;
    fs["xil"] >> xil;
    Rl = cv::Mat::eye(3, 3, CV_64F);
    fs["Rl"] >> Rl;
    fs["Kr"] >> Kr;
    fs["Dr"] >> Dr;
    fs["xir"] >> xir;
    fs["Rr"] >> Rr;
    fs["T"] >> Translation;
    fs["num_disparities"] >> num_disparities;
    fs["fps"] >> fps;
    fs["vfov"] >> virtual_fov;
    fs["max_depth"] >> max_depth;

    fs["enable_depth"] >> calculate_stereo;
    fs["enable_undistort"] >> enable_undistort;
    fs["use_cuda"] >> use_cuda;

    fs["disparity_window"] >> disparity_window;
    fs["pre_filter_cap"] >> pre_filter_cap;
    fs["min_disparity"] >> min_disparity;
    fs["texture_threshold"] >> texture_threshold;
    fs["uniqueness_ratio"] >> uniqueness_ratio;
    fs["speckle_window_size"] >> speckle_window_size;
    fs["speckle_range"] >> speckle_range;
    fs["disp12_max_diff"] >> disp12_max_diff;

    fs.release();

    image_width = cap_size.width / 2;
    image_height = cap_size.height;
    capture_width = cap_size.width;
    capture_height = cap_size.height;

    if (capture_width == 2560) {
        if (capture_height != 960) {
            std::cout << "Invalid size parameters: " << capture_width << ", " << capture_height << std::endl;
            exit(-1);
        }
    }
    else if (capture_width == 1280) {
        if (capture_height != 480) {
            std::cout << "Invalid size parameters: " << capture_width << ", " << capture_height << std::endl;
            exit(-1);
        }
    }
    else {
        std::cout << "Invalid size parameters: " << capture_width << ", " << capture_height << std::endl;
        exit(-1);
    }

    scale_ratio_w = image_width / cal_width;
    scale_ratio_h = image_height / cal_height;

    if (fps != 30 && fps != 20) {
        std::cout << "Invalid fps parameter: " << fps << std::endl;
        exit(-1);
    }
}

void StereoCalicam::init_undistort_rectify_map(cv::Mat K, cv::Mat D, cv::Mat xi, cv::Mat R,
                                               cv::Mat P, cv::Size size,
                                               cv::Mat &map1, cv::Mat &map2)
{
    map1 = cv::Mat(size, CV_32F);
    map2 = cv::Mat(size, CV_32F);

    double fx = scale_ratio_w * K.at<double>(0, 0);
    double fy = scale_ratio_h * K.at<double>(1, 1);
    double cx = scale_ratio_w * K.at<double>(0, 2);
    double cy = scale_ratio_h * K.at<double>(1, 2);
    double s = scale_ratio_w * K.at<double>(0, 1);

    double xid = xi.at<double>(0, 0);

    double k1 = scale_ratio_w * D.at<double>(0, 0);
    double k2 = scale_ratio_h * D.at<double>(0, 1);
    double p1 = scale_ratio_w * D.at<double>(0, 2);
    double p2 = scale_ratio_h * D.at<double>(0, 3);

    cv::Mat KRi = (P * R).inv();

    for (int r = 0; r < size.height; ++r)
    {
        for (int c = 0; c < size.width; ++c)
        {
            double xc = MatRowMul(KRi, c, r, 1., 0);
            double yc = MatRowMul(KRi, c, r, 1., 1);
            double zc = MatRowMul(KRi, c, r, 1., 2);

            double rr = sqrt(xc * xc + yc * yc + zc * zc);
            double xs = xc / rr;
            double ys = yc / rr;
            double zs = zc / rr;

            double xu = xs / (zs + xid);
            double yu = ys / (zs + xid);

            double r2 = xu * xu + yu * yu;
            double r4 = r2 * r2;
            double xd = (1 + k1 * r2 + k2 * r4) * xu + 2 * p1 * xu * yu + p2 * (r2 + 2 * xu * xu);
            double yd = (1 + k1 * r2 + k2 * r4) * yu + 2 * p2 * xu * yu + p1 * (r2 + 2 * yu * yu);

            double u = fx * xd + s * yd + cx;
            double v = fy * yd + cy;

            map1.at<float>(r, c) = (float)u;
            map2.at<float>(r, c) = (float)v;
        }
    }
}

void StereoCalicam::init_rectify_map()
{
    double vfov_rad = virtual_fov * CV_PI / 180.;
    double focal = image_height / 2. / tan(vfov_rad / 2.);
    Knew = (cv::Mat_<double>(3, 3) << focal, 0., image_width / 2. - 0.5,
            0., focal, image_height / 2. - 0.5,
            0., 0., 1.);

    cv::Size img_size(image_width, image_height);

    init_undistort_rectify_map(Kl, Dl, xil, Rl, Knew,
                               img_size, smap[0][0], smap[0][1]);

    std::cout << "Width: " << image_width << "\t"
              << "Height: " << image_height << "\t"
              << "V.Fov: " << virtual_fov << "\n";
    std::cout << "K Matrix: \n"
              << Knew << std::endl;

    init_undistort_rectify_map(Kr, Dr, xir, Rr, Knew,
                               img_size, smap[1][0], smap[1][1]);
    std::cout << "Ndisp: " << num_disparities << "\t";
    std::cout << std::endl;
}

void StereoCalicam::process()
{
    const auto t1 = std::chrono::system_clock::now();
    capture >> raw;
    raw(cv::Rect(0, 0, image_width, image_height)).copyTo(raw_left);
    raw(cv::Rect(image_width, 0, image_width, image_height)).copyTo(raw_right);

    if (enable_undistort) {
        cv::remap(raw_left, rect_left, smap[0][0], smap[0][1], 1, 0);
        cv::remap(raw_right, rect_right, smap[1][0], smap[1][1], 1, 0);
    }
    else {
        rect_left = raw_left;
        rect_right = raw_right;
    }

    if (calculate_stereo)
    {
#ifdef CUDA_STEREO_SGM
        if (use_cuda)
        {
            cv::cvtColor(rect_left, gray_left, cv::COLOR_BGR2GRAY);
            cv::cvtColor(rect_right, gray_right, cv::COLOR_BGR2GRAY);

            cudaMemcpy(d_I1->data, gray_left.data, input_bytes, cudaMemcpyHostToDevice);
            cudaMemcpy(d_I2->data, gray_right.data, input_bytes, cudaMemcpyHostToDevice);

            cuda_sgm->execute(d_I1->data, d_I2->data, d_disparity->data);
            cudaDeviceSynchronize();

            cudaMemcpy(disparity.data, d_disparity->data, output_bytes, cudaMemcpyDeviceToHost);
        }
        else
#endif  // CUDA_STEREO_SGM
        {
            cpu_sgbm->compute(rect_left, rect_right, disparity);
        }

        compute_depth();
    }
    const auto t2 = std::chrono::system_clock::now();
    
    const auto duration = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
    pipeline_fps = 1e6 / duration;
}


void StereoCalicam::compute_depth()
{
    double fx = Knew.at<double>(0, 0);
    double bl = -Translation.at<double>(0, 0);

    cv::Mat dispf;
    double conversion = 1.0f;
    if (!use_cuda) {
        conversion = 1.f / 16.f;
    }
    disparity.convertTo(dispf, CV_32F, conversion);
    for (int r = 0; r < dispf.rows; ++r)
    {
        for (int c = 0; c < dispf.cols; ++c)
        {
            double disp = dispf.at<float>(r, c);

            if (disp <= 0.0) {
                depth.at<float>(r, c) = 0.0f;
                continue;
            }
            depth.at<float>(r, c) = fx * bl / disp;
        }
    }
}

double StereoCalicam::get_distance(int x, int y, int radius)
{
    cv::Mat circle_mask = cv::Mat::zeros(depth.rows, depth.cols, CV_8UC1);
    cv::circle(circle_mask, cv::Size(x, y), radius, cv::Scalar(255, 255, 255), cv::FILLED);
    
    cv::Mat nonzero_mask = (depth > 0.1) & (depth < max_depth);
    nonzero_mask.convertTo(nonzero_mask, CV_8UC1);

    cv::Mat target_mask;
    cv::bitwise_and(circle_mask, nonzero_mask, target_mask);

    return cv::mean(depth, target_mask)[0];
}

cv::Mat StereoCalicam::get_debug_disparity()
{
    cv::Mat disparity_8u, disparity_color;
    disparity.convertTo(disparity_8u, CV_8U, 255. / num_disparities);
    cv::applyColorMap(disparity_8u, disparity_color, cv::COLORMAP_JET);
    disparity_color.setTo(cv::Scalar(0, 0, 0), disparity == invalid_disp);
    cv::putText(disparity_color, //target image
            format_string("%4.1f FPS", get_pipeline_fps()),
            cv::Point(50, 50), //top-left position
            cv::FONT_HERSHEY_DUPLEX,
            1.0,
            CV_RGB(255, 255, 255), //font color
            2);
    return disparity_color;
}

StereoCalicam::~StereoCalicam()
{
    capture.release();
}
