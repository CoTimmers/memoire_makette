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

        ecat::EL5102* encoder      = ecat_activity->continuous_state->encoder;
        ecat::EL1002* limit_switch = ecat_activity->continuous_state->limit_switch;
        ecat::EL7047* motor_x      = ecat_activity->continuous_state->motor_x;
        ecat::EL7047* motor_y      = ecat_activity->continuous_state->motor_y;
        ecat::EposDrive* motor_z   = ecat_activity->continuous_state->motor_z;

        motor_x->set_control_mode(EL7047_VELOCITY_CONTROL);
        motor_y->set_control_mode(EL7047_VELOCITY_CONTROL);
        motor_z->set_control_mode(EPOS_VELOCITY_CONTROL);
        motor_z->set_brake(0);
        motor_z->set_input(.0);

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
            {ecat_activity->discrete_state->gantry_state = activity_5c::ECATActivity::GANTRY_MOVE_TO_ATTACH;}
    }

    void move_to_attach(void *in)
    {
        activity_5c::ECATActivity* ecat_activity = (activity_5c::ECATActivity*) in;

        ecat::EL5102* encoder    = ecat_activity->continuous_state->encoder;
        ecat::EL7047* motor_x    = ecat_activity->continuous_state->motor_x;
        ecat::EL7047* motor_y    = ecat_activity->continuous_state->motor_y;
        ecat::EposDrive* motor_z = ecat_activity->continuous_state->motor_z;

        motor_x->set_control_mode(EL7047_VELOCITY_CONTROL);
        motor_y->set_control_mode(EL7047_VELOCITY_CONTROL);
        motor_z->set_control_mode(EPOS_VELOCITY_CONTROL);
        motor_z->set_brake(0);
        motor_z->set_input(.0);

/* Pas changer les vitesses pour commencer */
        double velocity_fast = 20;
        double velocity_slow = 1;
        double velocity_x, velocity_y;
        int move_complete_x = 0;
        int move_complete_y = 0;

/* choose values in ecat_activity_configuration_data.json */
        double target_x = ecat_activity->params->attach_position_x;
        double target_y = ecat_activity->params->attach_position_y;

/* 10mm avant d'arriver à la position finale */
        if(encoder->channel1->get_position() < target_x)
        {
            if(encoder->channel1->get_position() < target_x - 10)
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

        if(encoder->channel2->get_position() < target_y)
        {
            if(encoder->channel2->get_position() < target_y - 10)
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
            {ecat_activity->discrete_state->gantry_state = activity_5c::ECATActivity::GANTRY_WAITING_OPERATOR;}
    }

    void moving(void *in)
    {
        activity_5c::ECATActivity* ecat_activity = (activity_5c::ECATActivity*) in;

        ecat::EL5102* encoder    = ecat_activity->continuous_state->encoder;
        ecat::EL7047* motor_x    = ecat_activity->continuous_state->motor_x;
        ecat::EL7047* motor_y    = ecat_activity->continuous_state->motor_y;
        ecat::EposDrive* motor_z = ecat_activity->continuous_state->motor_z;

        motor_x->set_control_mode(EL7047_VELOCITY_CONTROL);
        motor_y->set_control_mode(EL7047_VELOCITY_CONTROL);
        motor_z->set_control_mode(EPOS_VELOCITY_CONTROL);
        motor_z->set_brake(0);
        motor_z->set_input(.0);

        /* TODO: implement move towards wall
           Waiting for pixels/mm calibration from promoter */
        ecat_activity->discrete_state->gantry_state = activity_5c::ECATActivity::GANTRY_WAITING_STEADY;
    }

    void waiting_operator(void *in)
    {
        activity_5c::ECATActivity* ecat_activity = (activity_5c::ECATActivity*) in;

        ecat_activity->continuous_state->motor_x->set_enable(0);
        ecat_activity->continuous_state->motor_x->set_input(0);
        ecat_activity->continuous_state->motor_y->set_enable(0);
        ecat_activity->continuous_state->motor_y->set_input(0);

        if(ecat_activity->discrete_state->OPERATOR_COMPLETE == 1)
        {
            ecat_activity->discrete_state->OPERATOR_COMPLETE = 0;
            ecat_activity->discrete_state->gantry_state = activity_5c::ECATActivity::GANTRY_WAITING_STEADY;
        }
    }

    void waiting_steady_state(void *in)
    {
        activity_5c::ECATActivity* ecat_activity = (activity_5c::ECATActivity*) in;

        ecat_activity->continuous_state->motor_x->set_enable(0);
        ecat_activity->continuous_state->motor_x->set_input(0);
        ecat_activity->continuous_state->motor_y->set_enable(0);
        ecat_activity->continuous_state->motor_y->set_input(0);

        if(!ecat_activity->continuous_state->marker_detected)
            {return;}

        double dx = ecat_activity->continuous_state->marker_pixel_x
                  - ecat_activity->continuous_state->marker_prev_pixel_x;
        double dy = ecat_activity->continuous_state->marker_pixel_y
                  - ecat_activity->continuous_state->marker_prev_pixel_y;
        double movement = sqrt(dx*dx + dy*dy);

        ecat_activity->continuous_state->marker_prev_pixel_x = ecat_activity->continuous_state->marker_pixel_x;
        ecat_activity->continuous_state->marker_prev_pixel_y = ecat_activity->continuous_state->marker_pixel_y;

/* Check if the marker is stable */
        if(movement < ecat_activity->params->steady_state_threshold_px)
            {ecat_activity->continuous_state->steady_state_frame_count++;}
        else
            {ecat_activity->continuous_state->steady_state_frame_count = 0;}

        if(ecat_activity->continuous_state->steady_state_frame_count >= ecat_activity->params->steady_state_n_frames)
        {
            ecat_activity->continuous_state->steady_state_frame_count = 0;
            ecat_activity->discrete_state->gantry_state = activity_5c::ECATActivity::GANTRY_CHECK_ALIGNMENT;
        }
    }

    void check_alignment(void *in)
    {
        activity_5c::ECATActivity* ecat_activity = (activity_5c::ECATActivity*) in;

        ecat_activity->continuous_state->motor_x->set_enable(0);
        ecat_activity->continuous_state->motor_x->set_input(0);
        ecat_activity->continuous_state->motor_y->set_enable(0);
        ecat_activity->continuous_state->motor_y->set_input(0);

        if(ecat_activity->continuous_state->wall_aligned)
            {ecat_activity->discrete_state->gantry_state = activity_5c::ECATActivity::GANTRY_DONE;}
        else
            {ecat_activity->discrete_state->gantry_state = activity_5c::ECATActivity::GANTRY_ADJUST_POSITION;}
    }

    void adjust_position(void *in)
    {
        activity_5c::ECATActivity* ecat_activity = (activity_5c::ECATActivity*) in;

        ecat::EL5102* encoder = ecat_activity->continuous_state->encoder;
        ecat::EL7047* motor_x = ecat_activity->continuous_state->motor_x;
        ecat::EL7047* motor_y = ecat_activity->continuous_state->motor_y;

        motor_x->set_control_mode(EL7047_VELOCITY_CONTROL);
        motor_y->set_control_mode(EL7047_VELOCITY_CONTROL);

        double velocity_fast = 20;
        double velocity_slow = 1;
        double target_x = ecat_activity->continuous_state->target_x;

        if(encoder->channel1->get_position() < target_x)
        {
            if(encoder->channel1->get_position() < target_x - 10)
                {motor_x->set_input(velocity_fast);}
            else
                {motor_x->set_input(velocity_slow);}
            motor_x->set_enable(1);
        }
        else
        {
            motor_x->set_enable(0);
            motor_x->set_input(0);
            ecat_activity->continuous_state->steady_state_frame_count = 0;
            ecat_activity->discrete_state->gantry_state = activity_5c::ECATActivity::GANTRY_WAITING_STEADY;
        }

        motor_y->set_enable(0);
        motor_y->set_input(0);
    }

    void done(void *in)
    {
        activity_5c::ECATActivity* ecat_activity = (activity_5c::ECATActivity*) in;

        ecat_activity->continuous_state->motor_x->set_enable(0);
        ecat_activity->continuous_state->motor_x->set_input(0);
        ecat_activity->continuous_state->motor_y->set_enable(0);
        ecat_activity->continuous_state->motor_y->set_input(0);
        ecat_activity->continuous_state->motor_z->set_input(.0);
    }
}
