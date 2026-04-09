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

#include "ethercat.h"

#include "ARNOLD_gantry/task_application.hpp"
#include "ARNOLD_gantry/ecat_activity.hpp"

#include "beckhoff_modules/EL7047.hpp"
#include "beckhoff_modules/EL5102.hpp"
#include "beckhoff_modules/EL1002.hpp"

#include "time_functions/time_functions.h"
#include "SOEM_helper_functions/etherCAT_communication.h"

#include <iostream>

using namespace std;

namespace ARNOLD{
    /* Semantic meaning of digital IO */
    int get_limit_switch_x(ecat::EL1002* read_module)
        {return !read_module->read_input1();}
    int get_limit_switch_y(ecat::EL1002* read_module)
        {return !read_module->read_input2();}
    
    

    /* Task functions */
    void do_nothing(void *in)
    {}

    void homing(void *in)
    {
        activity_5c::ECATActivity* ecat_activity = (activity_5c::ECATActivity*) in;

        ecat::EL5102* encoder = ecat_activity->continuous_state->encoder;
        ecat::EL1002* limit_switch = ecat_activity->continuous_state->limit_switch;
        ecat::EL7047* motor_x = ecat_activity->continuous_state->motor_x;
        ecat::EL7047* motor_y = ecat_activity->continuous_state->motor_y;
        ecat::EposDrive* motor_z = ecat_activity->continuous_state->motor_z;

        motor_x->set_control_mode(EL7047_VELOCITY_CONTROL);
        motor_y->set_control_mode(EL7047_VELOCITY_CONTROL);
        motor_z->set_control_mode(EPOS_VELOCITY_CONTROL);
        motor_z->set_brake(0);
        motor_z->set_input(.0);

        // int direction = -1;     // left
        int calibrated = 0;
        double velocity_fast = -8;
        double velocity_slow = -1;
        double velocity_x, velocity_y;
        int homing_complete_x = 0;
        int homing_complete_y = 0;

        if(get_limit_switch_x(limit_switch) == 1) 
        {
            /* Set reference pulse */ 
            encoder->channel1->set_counter(1);
            encoder->channel1->set_counter_value(0);

            motor_x->set_enable(0);
            motor_x->set_input(0);
            
            homing_complete_x = 1;
        }
        else
        {
            if(encoder->channel1->get_position() > 1.5)
                {velocity_x = velocity_fast;}
            else
                {velocity_x = velocity_slow;}
             
            motor_x->set_enable(1);
            motor_x->set_input(velocity_x);
        }
        if(get_limit_switch_y(limit_switch) == 1) 
        {
            /* Set reference pulse */ 
            encoder->channel2->set_counter(1);
            encoder->channel2->set_counter_value(0);

            motor_y->set_enable(0);
            motor_y->set_input(0);
            
            homing_complete_y = 1;
        }
        else
        {
            if(encoder->channel2->get_position() > 1.5)
                {velocity_y = velocity_fast;}
            else
                {velocity_y = velocity_slow;}
                
            motor_y->set_enable(1);
            motor_y->set_input(velocity_y);
        }

        if((homing_complete_x == 1) && (homing_complete_y == 1))
            {ecat_activity->discrete_state->HOMING_COMPLETE = 1;}
    }

    void move(void *in)
    {
        activity_5c::ECATActivity* ecat_activity = (activity_5c::ECATActivity*) in;

        ecat::EL5102* encoder = ecat_activity->continuous_state->encoder;
        ecat::EL7047* motor_x = ecat_activity->continuous_state->motor_x;
        ecat::EL7047* motor_y = ecat_activity->continuous_state->motor_y;
        ecat::EposDrive* motor_z = ecat_activity->continuous_state->motor_z;

        motor_x->set_control_mode(EL7047_VELOCITY_CONTROL);
        motor_y->set_control_mode(EL7047_VELOCITY_CONTROL);
        
        motor_z->set_control_mode(EPOS_VELOCITY_CONTROL);
        motor_z->set_brake(0);
        motor_z->set_input(-.0);

        double velocity_fast = 20;
        double velocity_slow = 1;
        double velocity_x, velocity_y;
        int move_complete_x = 0;
        int move_complete_y = 0;

        if(encoder->channel1->get_position() < 100)
        {
            if(encoder->channel1->get_position() < 98.5)
                {velocity_x = velocity_fast;}
            else
                {velocity_x = velocity_slow;}
             
            motor_x->set_enable(1);
            motor_x->set_input(velocity_x);
        }
        else
        {
            move_complete_x = 1;
            motor_x->set_enable(0);
            motor_x->set_input(0);
        }

        if(encoder->channel2->get_position() < 60)
        {
            if(encoder->channel2->get_position() < 58.5)
                {velocity_y = velocity_fast;}
            else
                {velocity_y = velocity_slow;}
             
            motor_y->set_enable(1);
            motor_y->set_input(velocity_y);
        }
        else
        {
            move_complete_y = 1;
            motor_y->set_enable(0);
            motor_y->set_input(0);
        }
        
        if((move_complete_x == 1) && (move_complete_y == 1))
            {ecat_activity->discrete_state->HOMING_COMPLETE = 0;}
    }
}

