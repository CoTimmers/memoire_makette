#ifndef IMAGE_PROCESSING_FUNCTIONS_HPP
#define IMAGE_PROCESSING_FUNCTIONS_HPP

#include <unistd.h>
#include <opencv2/opencv.hpp>

#include <threads.h>

namespace IP{   // Image Processing namespace
    int copy_image(cv::Mat input_frame, cv::Mat* output_frame);
    int crop_roi(cv::Mat input_frame, cv::Mat* output_frame, cv::Rect crop_window);
    int display_frame(char* window_name, cv::Mat* frame, int time);
    
    int colour_detection_hsv(cv::Mat input_frame, cv::Mat *output_frame, cv::Scalar lower_colour_threshold, cv::Scalar upper_colour_threshold);
    int median_blur_filter(cv::Mat input_frame, cv::Mat *output_frame, int filter_size);
    int calculate_centre_point_and_area(cv::Mat input_frame, cv::Point* point, double *area, double min_area);
    int transform_point_in_rect(cv::Point input_point, cv::Point *output_point, cv::Rect transform_window);
    int draw_circle_on_frame(cv::Mat input_frame, cv::Point centre, int radius, cv::Scalar colour);
    int draw_rectangle_on_frame(cv::Mat input_frame, cv::Rect rect, cv::Scalar colour, int thickness);
    int construct_rectangle_around_point(cv::Point point, cv::Rect* rect, int distance_from_centre);
    int fit_rectangle_in_frame(cv::Rect input_rect, cv::Rect* output_rect, cv::Mat frame);
 
    int read_undistortion_mappings(char* path_to_camera_parameters_file, cv::Mat* camera_matrix, cv::Mat* distortion_matrix);
    int undistort_point(cv::Point input, cv::Point* output, cv::Mat camera_matrix, cv::Mat distortion_matrix);
    int undistort_frame(cv::Mat input, cv::Mat* output, cv::Mat camera_matrix, cv::Mat distortion_matrix);
    int get_apriltag_data(cv::Mat input, cv::Mat* output);
    int save_image_frame(cv::Mat frame, char* file_path, double resize_factor, int image_quality);

    /* Bac detection */
    int detect_green_marker_centre(cv::Mat input_frame, cv::Point *centre);
    int detect_bac_contour(cv::Mat input_frame, cv::Point marker_centre,
                           std::vector<cv::Point> *corners);
}

#endif /* IMAGE_PROCESSING_FUNCTIONS_HPP */
