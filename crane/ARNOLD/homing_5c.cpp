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

// void print_IO(ecat::EL7047 *motor_x, ecat::EL5102 *encoder)
// {
//     cout << "DRIVE" << endl;
//     printf("enable %i\n", motor_x->get_enable());
//     // printf("ready_to_enable  %x\n", motor_x->ECAT->m_ECAT_outputs->control_status.ready_to_enable);
//     // printf("input %i\n", motor_x->ECAT->m_ECAT_inputs->input);
//     // printf("internal_position  %i\n", motor_x->ECAT->m_ECAT_outputs->internal_position);

//     cout << "ENCODER" << endl;
//     // cout << "counter_value: " << encoder->ECAT->m_ECAT_outputs->channel1.counter_value <<endl;
//     cout << "position: " << encoder->channel1->get_position() <<endl;
//     cout << endl;
// }

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

    ecat_activity->start_RT_thread();
    while(1)
    {
        if(*KILLSWITCH_ENGAGED == 1)
            {deintialisation_request = 1;}
    }

    // start_homing(ecat_activity);
}

