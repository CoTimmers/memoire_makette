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
            cout << "Attach crate to hook and press Enter..." << endl;
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

        /* Detect bac contour and check alignment */
        if(ecat_activity->discrete_state->gantry_state == activity_5c::ECATActivity::GANTRY_CHECK_ALIGNMENT)
        {
            if(detected == 1)
            {
                vector<cv::Point> corners;
                int bac_found = IP::detect_bac_contour(frame, marker_centre, &corners);

                if(bac_found == 1)
                {
                    /* TODO: compare corners with hardcoded wall position */
                    /* Waiting for pixels/mm calibration */
                    ecat_activity->continuous_state->wall_aligned = 0;
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
