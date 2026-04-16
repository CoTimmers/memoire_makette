#include <iostream>
#include <sstream>
#include <cstdio>
#include <unistd.h>

extern "C" {
#include "apriltag.h"
#include "tag36h11.h"
#include "tag25h9.h"
#include "tag16h5.h"
#include "tagCircle21h7.h"
#include "tagCircle49h12.h"
#include "tagCustom48h12.h"
#include "tagStandard41h12.h"
#include "tagStandard52h13.h"
#include "common/getopt.h"

#include "apriltag_pose.h"
}

#include <Eigen/Geometry>

#include "image_processing_activity/image_processing_functions.hpp"

using namespace std;

namespace IP{
    /* Functions */
    /* opencv wrapper */
    int copy_image(cv::Mat input_frame, cv::Mat* output_frame)
    {
        *output_frame = input_frame.clone();
        // cv::imshow("test", input_frame);
        // cv::waitKey(1);
        
        return 1;
    }

    /* Cuts the part defined by crop_window out of input frame, and returns it with output_frame */
    int crop_roi(cv::Mat input_frame, cv::Mat* output_frame, cv::Rect crop_window)
    {
        cv::Rect image_area(0,0,input_frame.cols,input_frame.rows);
        
        int left_boundary = crop_window.x;
        int top_boundary = crop_window.y;
        int right_boundary = crop_window.x + crop_window.width;
        int bottom_boundary = crop_window.y + crop_window.height;

        /* Check if crop roi is inside image */
        if((left_boundary < 0) || (left_boundary >= image_area.size().width))
            {return 0;}
            // {cout << "WARNING: left boundary out of frame" << endl;}
        else if((right_boundary < left_boundary) || (right_boundary >= image_area.size().width))
            {return 0;}
            // {cout << "WARNING: right boundary out of frame" << endl;}
        else if((top_boundary < 0) || (top_boundary >= image_area.size().height))
            {return 0;}
            // {cout << "WARNING: top boundary out of frame" << endl;}
        else if((bottom_boundary < top_boundary) || (bottom_boundary >= image_area.size().height))
            {return 0;}
            // {cout << "WARNING: bottom boundary out of frame" << endl;}
        else
            {*output_frame = input_frame(crop_window).clone();}

        return 1;
    }
    
    /* Displays frame in a window with window_name name, holds for time (ms) amount */
    int display_frame(char* window_name, cv::Mat* frame, int time)
    {
        cv::imshow(window_name, *frame);
        cv::waitKey(time);

        return 1;
    }
    
    // int resize_frame() // TODO

    /* Returns 0 if resize is not possible*/
    int save_image_frame(cv::Mat frame, char* file_path, double resize_factor, int image_quality)
    {
        /* Only execute function if resize gives a valid frame, i.e. at least 1x1 px */
        if((frame.rows*resize_factor <= 0.5) || (frame.cols*resize_factor <= 0.5)) // 0.5 since this gets rounded later on 
            {return 0;} 
        
        cv::Mat temp;
        cv::resize(frame, temp, cv::Size(), resize_factor, resize_factor);
        cv::imwrite("temp.jpg", temp, {cv::IMWRITE_JPEG_QUALITY, image_quality});
        std::rename("temp.jpg", file_path);
        std::remove("temp.jpg");
        return 1;
    }

    /* Turns input frame to hsv and does a colour thresholding */
    int colour_detection_hsv(cv::Mat input_frame, cv::Mat *output_frame, cv::Scalar lower_colour_threshold, cv::Scalar upper_colour_threshold)
    {
        cv::cvtColor(input_frame, *output_frame,  cv::COLOR_BGR2HSV);
        if(lower_colour_threshold(0) < 0)      // Wrap around hue
        {
            cv::Mat mask1, mask2;
            cv::Scalar lower_bound1 = cv::Scalar(180+lower_colour_threshold(0),lower_colour_threshold(1),lower_colour_threshold(2));
            cv::Scalar upper_bound1 = cv::Scalar(180,upper_colour_threshold(1),upper_colour_threshold(2));
            cv::Scalar lower_bound2 = cv::Scalar(0,lower_colour_threshold(1),lower_colour_threshold(2));
            cv::Scalar upper_bound2 = upper_colour_threshold;

            cv::inRange(*output_frame, lower_bound1, upper_bound1, mask1);
            cv::inRange(*output_frame, lower_bound2, upper_bound2, mask2);
            cv::bitwise_or(mask1,mask2,*output_frame);
        }
        else
        {
            cv::inRange(*output_frame, lower_colour_threshold, upper_colour_threshold, *output_frame);
        }
        
        return 1;
    }

    /* Wrapper for opencv function */
    int median_blur_filter(cv::Mat input_frame, cv::Mat *output_frame, int filter_size)
    {
        cv::medianBlur(input_frame,*output_frame,filter_size);

        return 1;
    }

    /* Calculates the centre point of input frame, based on its moments, returns area of pixels as well */
    int calculate_centre_point_and_area(cv::Mat input_frame, cv::Point* point, double *area, double min_area)
    {
        cv::Moments mts = cv::moments(input_frame,true); 

        double m01 = mts.m01;
        double m10 = mts.m10;
        *area = mts.m00;
        if(mts.m00 >= min_area) // Only update if area if larger than min area
        {
            double x_pos = m10/(*area);
            double y_pos = m01/(*area);

            *point = cv::Point(x_pos,y_pos);
        }
        else
            {return 0;}

        return 1;
    }

    int transform_point_in_rect(cv::Point input_point, cv::Point *output_point, cv::Rect transform_window)
    {
        *output_point = input_point + transform_window.tl();
        return 1;
    }

    /* Wrapper for opencv function */
    int draw_circle_on_frame(cv::Mat input_frame, cv::Point centre, int radius, cv::Scalar colour)
    {
        cv::circle(input_frame, centre, radius, colour, -1);
        return 1;
    }

    /* Wrapper for opencv function */
    int draw_rectangle_on_frame(cv::Mat input_frame, cv::Rect rect, cv::Scalar colour, int thickness)
    {
        cv::rectangle(input_frame, rect, colour, thickness);
        return 1;
    }

    int construct_rectangle_around_point(cv::Point point, cv::Rect* rect, int distance_from_centre) 
    {   
        cv::Rect temp = cv::Rect(cv::Point(point.x-distance_from_centre,point.y-distance_from_centre), 
                                 cv::Point(point.x+distance_from_centre,point.y+distance_from_centre));
        *rect = temp;
        return 1;
    }

    int fit_rectangle_in_frame(cv::Rect input_rect, cv::Rect* output_rect, cv::Mat frame)
    {
        /* Check boundaries */
        cv::Point tl = input_rect.tl();
        cv::Point br = input_rect.br();

        if(tl.x < 0)
            {tl.x = 0;}
        if(br.x > frame.cols-1)   
            {br.x = frame.cols-1;}
        if(tl.y < 0)
            {tl.y = 0;}
        if(br.y > frame.rows-1)
            {br.y = frame.rows-1;}

        *output_rect = cv::Rect(tl,br);

        return 1;
    }

    /* Returns the camera_matrix of undistorted camera, and the distortion matrix */
    int read_undistortion_mappings(char* path_to_camera_parameters_file, cv::Mat* camera_matrix, cv::Mat* distortion_matrix)
    {
        /* Camera parameters */
        string path_string = path_to_camera_parameters_file;
        cv::FileStorage fs(path_string, cv::FileStorage::READ); // Read the settings
        if (!fs.isOpened())
        {
            cout << "Could not open the undistortion mapping file: \"" << path_to_camera_parameters_file << "\"" << endl;
            return 0;                
        }
        else
        {
            cv::Mat t_camera_mat, t_distortion_mat;
            fs["camera_matrix"] >> t_camera_mat;
            fs["distortion_coefficients"] >> t_distortion_mat;
            fs.release();                                         // close Settings file

            *distortion_matrix = t_distortion_mat.clone();
            *camera_matrix = t_camera_mat.clone();
        }
        
        return 1;
    }

/* make this work */
    int undistort_point(cv::Point input, cv::Point* output, cv::Mat camera_matrix, cv::Mat distortion_matrix)
    {
        cv::Mat_<cv::Point2f> points(1,1);
        points(0) = cv::Point2f(input.x,input.y);

        cv::Mat R, temp;
        cv::undistortPoints(points, temp, camera_matrix, distortion_matrix, R, camera_matrix);

        output->x = temp.at<float>(0);
        output->y = temp.at<float>(1);

        return 1;
    }
    
    int undistort_frame(cv::Mat input, cv::Mat* output, cv::Mat camera_matrix, cv::Mat distortion_matrix)
    {
        cv::Mat temp;
        cv::undistort(input,temp,camera_matrix,distortion_matrix,camera_matrix);
        
        *output = temp.clone();

        return 1;
    }

    int get_apriltag_data(cv::Mat input, cv::Mat* output)
    {
        cv::Mat temp = input.clone();
        apriltag_family_t *tf = NULL;
        tf = tagStandard41h12_create();

        apriltag_detector_t *td = apriltag_detector_create();
        apriltag_detector_add_family(td, tf);

        apriltag_detection_info_t info;
        info.tagsize = 0.111;
        info.fx = 1188.80430681;
        info.fy = 1188.80430681;
        info.cx = 959.5;
        info.cy = 599.5;

        cv::Mat gray;
        cvtColor(temp, gray, cv::COLOR_BGR2GRAY);
        // Make an image_u8_t header for the Mat data
        image_u8_t im = { .width = gray.cols,
            .height = gray.rows,
            .stride = gray.cols,
            .buf = gray.data
        };
        zarray_t *detections = apriltag_detector_detect(td, &im);

        apriltag_pose_t pose;

        apriltag_detection_t *det;
        if(zarray_size(detections) != 0)
        {
            zarray_get(detections, 0, &det);

            // dRAW
            cv::line(temp, cv::Point(det->p[0][0], det->p[0][1]),
                cv::Point(det->p[1][0], det->p[1][1]), cv::Scalar(0, 0xff, 0), 2);
            cv::line(temp, cv::Point(det->p[0][0], det->p[0][1]),
                    cv::Point(det->p[3][0], det->p[3][1]), cv::Scalar(0, 0, 0xff), 2);
            cv::line(temp, cv::Point(det->p[1][0], det->p[1][1]),
                    cv::Point(det->p[2][0], det->p[2][1]), cv::Scalar(0xff, 0, 0), 2);
            cv::line(temp, cv::Point(det->p[2][0], det->p[2][1]),
                    cv::Point(det->p[3][0], det->p[3][1]), cv::Scalar(0xff, 0, 0), 2);
            
            // Text
            stringstream ss;
            ss << det->id;
            cv::String text = ss.str();
            int fontface = cv::FONT_HERSHEY_SCRIPT_SIMPLEX;
            double fontscale = 1.0;
            int baseline;
            cv::Size textsize = cv::getTextSize(text, fontface, fontscale, 2, &baseline);
            putText(temp, text, cv::Point(det->c[0]-textsize.width/2, 
                det->c[1]+textsize.height/2),fontface, fontscale, cv::Scalar(0xff, 0x99, 0), 2);
            
            // Then call estimate_tag_pose.
            info.det = det;
            double err = estimate_tag_pose(&info, &pose);

            Eigen::Matrix<double,3,3> mat = Eigen::Map<Eigen::Matrix<double,3,3>>(pose.R->data);
            Eigen::Matrix<double,3,1> euler_angles = mat.eulerAngles(2, 1, 0); 

            double position[3];
            position[0] = pose.t->data[0];
            position[1] = pose.t->data[1];
            position[2] = pose.t->data[2];

            double rotation[3];
            rotation[0] = euler_angles[0]/M_PI*180;
            rotation[1] = euler_angles[1]/M_PI*180;
            rotation[2] = euler_angles[2]/M_PI*180;

            printf("Position: %lf  %lf  %lf\n",position[0], position[1], position[2]);
            printf("Rotation: %lf  %lf  %lf\n   ",rotation[0], rotation[1], rotation[2]);
        }
        
        apriltag_detections_destroy(detections);
        
        apriltag_detector_destroy(td);
        tag36h11_destroy(tf);

        *output = temp.clone();

        return 1;
    }
//     int ImageProcessingData::detect_clusters(cv::Mat mask, int number_of_trackers)
//     {
//         cv::TermCriteria criteria = cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 10, 1.0);
//         std::vector<cv::Point2f> centers;
//         cv::Mat labels;

//         cv::Mat x,y;
//         for ( int i=0; i<mask.rows; i++ )
//         {
//             for ( int j=0; j<mask.cols; j++ )
//             {
//                 if ( mask.at<uchar>(i,j) == 255)
//                 {
//                     x.push_back(i);
//                     y.push_back(j);
//                 }
//             }
//         }
//         x.convertTo(x , CV_32F);
//         y.convertTo(y , CV_32F);
//         cv::Mat coords(x.rows*x.cols, 1 , CV_32FC2);
//         for(int i=0; i<x.rows; i++)
//         {
//             coords.at<float>(i,1) = x.at<float>(i,0);
//             coords.at<float>(i,0) = y.at<float>(i,0);
//         }
//         cv::kmeans(coords,number_of_trackers,labels,criteria,10,cv::KMEANS_RANDOM_CENTERS,centers);

//         for(int i = 0; i < number_of_trackers; i++)
//         {
//             centers[i].x += m_home_window.tl().x-m_workspace.tl().x;
//             centers[i].y += m_home_window.tl().y-m_workspace.tl().y;
//         }

//         if(number_of_trackers > 1)
//         {
//             if(centers[1].y > centers[0].y) 
//             {
//                 m_window_ee = track_window(centers[0]);
//                 m_window_load = track_window(centers[1]);
//             }
//             else
//             {
//                 m_window_ee = track_window(centers[1]);
//                 m_window_load = track_window(centers[0]);
//             }
//         }
//         else
//             {m_window_ee = track_window(centers[0]);}
//     }

//    
//     cv::Point ImageProcessingData::UndistortPoint(cv::Point input)
//     {
//         cv::Point output;
//         cv::Mat_<cv::Point2f> points(1,1);
//         points(0) = cv::Point2f(input.x,input.y);
//         pthread_mutex_lock(m_mutex);

//         cv::Mat R, temp;
//         cv::undistortPoints(points, temp, m_camera_mat, m_distortion_mat, R, m_camera_mat);

//         pthread_mutex_unlock(m_mutex);

//         output.x = temp.at<float>(0);
//         output.y = temp.at<float>(1);

//         return output;
//     }

//     cv::Rect ImageProcessingData::track_window(cv::Point track_position_px)
//     {   
//         cv::Rect window = cv::Rect(cv::Point(track_position_px.x-m_frame_size/2,track_position_px.y-m_frame_size/2), 
//             cv::Point(track_position_px.x+m_frame_size/2,track_position_px.y+m_frame_size/2));

//         /* Check boundaries */
//         cv::Point tl = window.tl();
//         cv::Point br = window.br();

//         if(tl.x < 0){  
//             tl.x = 0;}
//         if(br.x > m_frame.cols-1){
//             br.x = m_frame.cols-1;}
//         if(tl.y < 0){ 
//             tl.y = 0;}
//         if(br.y > m_frame.rows-1){
//             br.y = m_frame.rows-1;}

//         pthread_mutex_lock(m_mutex);
//         cv::Rect window2 = cv::Rect(tl,br);
//         pthread_mutex_unlock(m_mutex);
//         return window2;
//     }

//     /* <\ImageProcessingData> */

    /* ------------------------------------------------------------------ */
    /* detect_green_marker_centre                                           */
    /*                                                                      */
    /* Detects the green square marker in the image and returns its centre. */
    /*                                                                      */
    /* Parameters:                                                          */
    /*   input_frame  — BGR image from camera                               */
    /*   centre       — output: pixel coordinates of green marker centre    */
    /*                                                                      */
    /* Returns 1 if marker found, 0 if not found.                          */
    /* ------------------------------------------------------------------ */
    int detect_green_marker_centre(cv::Mat input_frame, cv::Point *centre)
    {
        /* Step 1 — Convert to HSV and threshold for green */
        /* Tune these values if lighting changes:
           Hue 40-80 covers most greens
           Saturation > 80 excludes grey/white
           Value > 80 excludes dark areas */
        cv::Scalar lower_green(40, 80, 80);
        cv::Scalar upper_green(80, 255, 255);

        cv::Mat green_mask;
        colour_detection_hsv(input_frame, &green_mask, lower_green, upper_green);

        /* Step 2 — Blur to remove noise */
        cv::Mat blurred;
        median_blur_filter(green_mask, &blurred, 5);

        /* Step 3 — Calculate centre of green region */
        double area;
        const double MIN_GREEN_AREA = 200.0;  // minimum pixels to be a valid marker
        if(calculate_centre_point_and_area(blurred, centre, &area, MIN_GREEN_AREA) != 1)
        {
            return 0;  // marker not found
        }

        return 1;
    }

    /* ------------------------------------------------------------------ */
    /* detect_bac_contour                                                   */
    /*                                                                      */
    /* Finds the red rectangular contour that contains the green marker.   */
    /* This identifies the correct bac when multiple red objects exist.    */
    /*                                                                      */
    /* Parameters:                                                          */
    /*   input_frame    — BGR image from camera                             */
    /*   marker_centre  — centre of green marker (from detect_green_marker) */
    /*   corners        — output: 4 corner points of the bac contour        */
    /*                    ordered: top-left, top-right, bottom-right,       */
    /*                             bottom-left                              */
    /*                                                                      */
    /* Returns 1 if bac found, 0 if not found.                             */
    /* ------------------------------------------------------------------ */
    int detect_bac_contour(cv::Mat input_frame, cv::Point marker_centre,
                           std::vector<cv::Point> *corners)
    {
        /* Step 1 — Threshold for red
           Calibrated on real Stella Artois bac under lab lighting.
           The red splits across both ends of the HSV hue circle:
             - low red  : hue 0-15   (~5900 pixels sampled)
             - high red : hue 165-180 (~11000 pixels sampled)
           Saturation minimum 150 (mean ~230 on real bac).
           Value minimum 80 to handle shadowed areas. */
        cv::Scalar lower_red1(0,   150, 80);
        cv::Scalar upper_red1(15,  255, 255);
        cv::Scalar lower_red2(165, 150, 80);
        cv::Scalar upper_red2(180, 255, 255);

        cv::Mat red_mask1, red_mask2, red_mask;
        colour_detection_hsv(input_frame, &red_mask1, lower_red1, upper_red1);
        colour_detection_hsv(input_frame, &red_mask2, lower_red2, upper_red2);
        cv::bitwise_or(red_mask1, red_mask2, red_mask);

        /* Step 2 — Blur to remove noise */
        cv::Mat blurred;
        median_blur_filter(red_mask, &blurred, 5);

        /* Step 3 — Find all contours in the red mask */
        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(blurred, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

        if(contours.empty())
        {
            std::cout << "[detect_bac_contour] Aucun contour rouge trouvé." << std::endl;
            return 0;
        }

        /* Step 4 — Find the contour that contains the green marker centre */
        for(auto &contour : contours)
        {
            /* Skip contours that are too small to be a bac */
            double area = cv::contourArea(contour);
            if(area < 5000.0)  // tune this threshold based on your camera height
                {continue;}

            /* Check if marker centre is inside this contour */
            double inside = cv::pointPolygonTest(contour, cv::Point2f(marker_centre), false);
            if(inside < 0)
                {continue;}  // marker not inside this contour

            /* Step 5 — Approximate contour to a rectangle (4 corners) */
            std::vector<cv::Point> approx;
            double epsilon = 0.02 * cv::arcLength(contour, true);
            cv::approxPolyDP(contour, approx, epsilon, true);

            /* We expect 4 corners for a rectangle */
            if(approx.size() != 4)
            {
                /* If approxPolyDP doesn't give exactly 4, use bounding rotated rect */
                cv::RotatedRect rect = cv::minAreaRect(contour);
                cv::Point2f rect_points[4];
                rect.points(rect_points);

                corners->clear();
                for(int i = 0; i < 4; i++)
                    {corners->push_back(cv::Point(rect_points[i]));}
            }
            else
            {
                *corners = approx;
            }

            /* Order corners: top-left, top-right, bottom-right, bottom-left */
            /* Sort by Y first (top vs bottom), then by X (left vs right) */
            std::sort(corners->begin(), corners->end(),
                [](const cv::Point &a, const cv::Point &b)
                    {return a.y < b.y;});

            /* Top two points — sort by X */
            if((*corners)[0].x > (*corners)[1].x)
                {std::swap((*corners)[0], (*corners)[1]);}
            /* Bottom two points — sort by X */
            if((*corners)[2].x > (*corners)[3].x)
                {std::swap((*corners)[3], (*corners)[2]);}

            std::cout << "[detect_bac_contour] Bac trouvé. Coins:" << std::endl;
            std::cout << "  top-left     : " << (*corners)[0] << std::endl;
            std::cout << "  top-right    : " << (*corners)[1] << std::endl;
            std::cout << "  bottom-right : " << (*corners)[2] << std::endl;
            std::cout << "  bottom-left  : " << (*corners)[3] << std::endl;

            return 1;
        }

        std::cout << "[detect_bac_contour] Aucun bac rouge contenant le marqueur." << std::endl;
        return 0;
    }

} // namespace IP

