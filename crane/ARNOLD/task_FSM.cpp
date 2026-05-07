/**
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
#include <iostream>
using namespace std;

namespace ARNOLD{

    void homing_FSM(void *in)
    {
        activity_5c::ECATActivity* ecat_activity = (activity_5c::ECATActivity*) in;
        
        if(ecat_activity->discrete_state->HOMING_COMPLETE == 0)
            {ecat_activity->continuous_state->execute_task_function = &homing;}
        else if(ecat_activity->discrete_state->HOMING_COMPLETE == 1)
            {ecat_activity->continuous_state->execute_task_function = &move;}
        
        ecat::EL5102* encoder    = ecat_activity->continuous_state->encoder;
        ecat::EL7047* motor_x    = ecat_activity->continuous_state->motor_x;
        ecat::EL7047* motor_y    = ecat_activity->continuous_state->motor_y;
        ecat::EposDrive* motor_z = ecat_activity->continuous_state->motor_z;
        cout << "ENCODER" << endl;
        cout << "position_x: " << encoder->channel1->get_position() << endl;
        cout << "position_y: " << encoder->channel2->get_position() << endl;
        cout << "velocity z: " << motor_z->get_velocity() << endl;
        cout << "input z: "    << motor_z->get_input() << endl;
    }

    void gantry_FSM(void *in)
    {
        activity_5c::ECATActivity* ecat_activity = (activity_5c::ECATActivity*) in;

        if(ecat_activity->discrete_state->HOMING_COMPLETE == 0)
        {
            cout << "[GANTRY_FSM] Homing non complété, FSM bloqué." << endl;
            return;
        }

        if(ecat_activity->discrete_state->gantry_state == activity_5c::ECATActivity::GANTRY_WAITING_OPERATOR)
            {ecat_activity->continuous_state->execute_task_function = &waiting_operator;}
        else if(ecat_activity->discrete_state->gantry_state == activity_5c::ECATActivity::GANTRY_WAITING_STEADY)
            {ecat_activity->continuous_state->execute_task_function = &waiting_steady_state;}
        else if(ecat_activity->discrete_state->gantry_state == activity_5c::ECATActivity::GANTRY_MOVING)
            {ecat_activity->continuous_state->execute_task_function = &moving;}
        else if(ecat_activity->discrete_state->gantry_state == activity_5c::ECATActivity::GANTRY_CHECK_ALIGNMENT)
            {ecat_activity->continuous_state->execute_task_function = &check_alignment;}
        else if(ecat_activity->discrete_state->gantry_state == activity_5c::ECATActivity::GANTRY_ADJUST_POSITION)
            {ecat_activity->continuous_state->execute_task_function = &adjust_position;}
        else if(ecat_activity->discrete_state->gantry_state == activity_5c::ECATActivity::GANTRY_DONE)
            {ecat_activity->continuous_state->execute_task_function = &done;}

        ecat::EL5102* encoder    = ecat_activity->continuous_state->encoder;
        ecat::EL7047* motor_x    = ecat_activity->continuous_state->motor_x;
        ecat::EL7047* motor_y    = ecat_activity->continuous_state->motor_y;
        ecat::EposDrive* motor_z = ecat_activity->continuous_state->motor_z;
        cout << "ENCODER" << endl;
        cout << "[GANTRY_FSM] state:        " << ecat_activity->discrete_state->gantry_state << endl;
        cout << "[GANTRY_FSM] FLAG_ALIGNED: " << ecat_activity->discrete_state->FLAG_ALIGNED << endl;
        cout << "position_x: " << encoder->channel1->get_position() << endl;
        cout << "position_y: " << encoder->channel2->get_position() << endl;
        cout << "velocity z: " << motor_z->get_velocity() << endl;
        cout << "input z: "    << motor_z->get_input() << endl;
    }
}