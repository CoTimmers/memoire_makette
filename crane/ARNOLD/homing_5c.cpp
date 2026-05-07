/** 
 * Example code to run and test the STEP Motor Drives with SOEM 
 * 
 * Author: Boris Deroo 
 * KU Leuven, department of Mechanical Engineering 
 * ROB Group
 */

#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <unistd.h>
#include <sched.h>
#include <string.h>
#include <time.h>
#include <pthread.h>
#include <math.h>
#include <signal.h>

#include <thread>

#include "ethercat.h"
#include <read_file/read_file.h>
#include "time_functions/time_functions.h"

/* 5C */
#include <five_c/activity/activity.h>
#include <five_c/thread/thread.h>

#include <ARNOLD_gantry/ecat_activity.hpp>

/* Camera thrread*/
#include <opencv2/opencv.hpp>
#include "image_processing_5C/include/image_processing_activity/image_processing_functions.hpp"

#include <iostream>

using namespace std;

int execute_application_thread = 1; // Runs thread defined here
bool deintialisation_request = 0;    // Linked to activity
bool configuration_request = 0;
bool execution_request = 1;

void signalHandler( int signum ) 
{
    cout << "Interrupt signal (" << signum << ") received.\n";
    
    /* Close down SOEM connection */
    execute_application_thread = 0;
    deintialisation_request = 1;

    usleep(1000000);
    printf("Ending program\n");
    exit(signum);  
}

void operator_thread(activity_5c::ECATActivity* ecat_activity)
{
    while(execute_application_thread)
    {
        if(ecat_activity->discrete_state->gantry_state == activity_5c::ECATActivity::GANTRY_WAITING_OPERATOR)
        {
            cout << "Attach crate to hook and press Enter" << endl;
            cin.get();
            ecat_activity->discrete_state->OPERATOR_COMPLETE = 1;
        }
        usleep(100000);
    }
}

void camera_thread(activity_5c::ECATActivity* ecat_activity)
{
    cv::VideoCapture cap(0);
    if(!cap.isOpened())
        {return;}

    /* HSV thresholds for green marker */
    cv::Scalar lower_green(40, 80, 80);
    cv::Scalar upper_green(80, 255, 255);

    while(execute_application_thread)
    {
        cv::Mat frame;
        cap >> frame;
        if(frame.empty())
            {usleep(10000); continue;}

        /* Detect green marker */
        cv::Point marker_centre;
        int detected = IP::detect_green_marker_centre(frame, &marker_centre);

        ecat_activity->continuous_state->marker_detected  = (detected == 1);
        if(detected == 1)
        {
            ecat_activity->continuous_state->marker_pixel_x = marker_centre.x;
            ecat_activity->continuous_state->marker_pixel_y = marker_centre.y;
        }

        if(ecat_activity->discrete_state->gantry_state ==
           activity_5c::ECATActivity::GANTRY_CHECK_ALIGNMENT)
        {
            cv::Mat hsv;
            cv::cvtColor(frame, hsv, cv::COLOR_BGR2HSV);

            cv::Mat mask1, mask2, mask_red;
            cv::inRange(hsv, cv::Scalar(0,   100, 100), cv::Scalar(10,  255, 255), mask1);
            cv::inRange(hsv, cv::Scalar(160, 100, 100), cv::Scalar(180, 255, 255), mask2);
            cv::bitwise_or(mask1, mask2, mask_red);

            vector<vector<cv::Point>> contours;
            cv::findContours(mask_red, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

            ecat_activity->discrete_state->FLAG_ALIGNED = 0;

            if(!contours.empty())
            {
                int largest = 0;
                for(int i = 1; i < (int)contours.size(); i++)
                    if(cv::contourArea(contours[i]) > cv::contourArea(contours[largest]))
                        largest = i;

                cv::RotatedRect rect = cv::minAreaRect(contours[largest]);
                cv::Point2f corners[4];
                rect.points(corners);

                double x_wall   = ecat_activity->params->x_wall_px;
                double eps_wall = ecat_activity->params->epsilon_wall_px;

                vector<cv::Point2f> wall_corners;
                for(int i = 0; i < 4; i++)
                {
                    if(fabs(corners[i].x - x_wall) <= eps_wall)
                        wall_corners.push_back(corners[i]);
                }

                if(wall_corners.size() >= 2)
                {
                    double D = sqrt(pow(wall_corners[0].x - wall_corners[1].x, 2.0) +
                                    pow(wall_corners[0].y - wall_corners[1].y, 2.0));
                    (void)D;
                    ecat_activity->discrete_state->FLAG_ALIGNED = 1;
                }
            }
        }

        usleep(33000);  // ~30 fps
    }

    cap.release();
}

int main()
{
    signal(SIGINT, signalHandler);
    activity_5c::ECATActivity* ecat_activity = new activity_5c::ECATActivity;

    strcpy(ecat_activity->params->configuration_file, "../configuration_files/ecat_activity_configuration_data.json");

    /* Share Memory */
    ecat_activity->coordination_state->deinitialisation_request = &deintialisation_request;
    ecat_activity->coordination_state->execution_request = &execution_request;
    ecat_activity->coordination_state->configuration_request = &configuration_request;
    int *KILLSWITCH_ENGAGED = &(ecat_activity->continuous_state->motor_z->flags.FAULT_STATE);

    /* Run Threads */
    thread_t thread_ecat_activity;
    int ecat_activity_cycletime = 1;    // Cycletime in ms
    create_thread(&thread_ecat_activity, (char*) "thread_ecat_activity", ecat_activity_cycletime);
    register_activity(&thread_ecat_activity, ecat_activity->activity, (char*) "ecat_activity");

    pthread_t ecat_activity_5cthread;
    pthread_create(&ecat_activity_5cthread, NULL, do_thread_loop, ((void*) &thread_ecat_activity));

    /* Camera and operator threads */
    thread camera(camera_thread, ecat_activity);
    camera.detach();

    thread operateur(operator_thread, ecat_activity);
    operateur.detach();

    ecat_activity->start_RT_thread();
    while(1)
    {
        if(*KILLSWITCH_ENGAGED == 1)
            {deintialisation_request = 1;}
    }
}
