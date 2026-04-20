/* ----------------------------------------------------------------------------
 * Project Title,
 * ROB @ KU Leuven, Leuven, Belgium
 * See LICENSE for the license information
 * -------------------------------------------------------------------------- */

/**
 * @file ecat_activity.cpp
 * @date June 6, 2023
 **/

#include <stdio.h>
#include <string.h>
#include <inttypes.h>
#include <unistd.h>
#include <stdlib.h>
#include <iostream>
#include <math.h>       
#include <thread>       

#include "ethercat.h"   

#include <beckhoff_modules/EL1002.hpp>
#include <beckhoff_modules/EL5102.hpp>
#include <beckhoff_modules/EL7047.hpp>

#include "epos_SOEM_driver/epos_driver.hpp"

#include <five_c/activity/activity.h>
#include <read_file/read_file.h>
#include <SOEM_helper_functions/etherCAT_communication.h>
#include "time_functions/time_functions.h"

#include "ARNOLD_gantry/ecat_activity.hpp"
#include "ARNOLD_gantry/task_application.hpp"
#include "ARNOLD_gantry/task_FSM.hpp"

namespace activity_5c
{
    /* This might not belong here */
    void ECATActivity::RT_thread()
    {
        int64_t time_offset = 0;  
        struct timespec ts;
        clock_gettime(CLOCK_MONOTONIC,&ts);

        int sync_offset_ar[100];  
        int average_sync_offset;  
        for(int i = 0; i < sizeof(sync_offset_ar)/sizeof(sync_offset_ar[0]); i++)
            {sync_offset_ar[i] = 1000000;}

        while(discrete_state->EXECUTE_RT_THREAD == 1)
        {   
            ec_send_processdata();
            ec_receive_processdata(EC_TIMEOUTRET);
            
            add_timespec(&ts, params->RT_thread_cycletime + time_offset);   
            clock_nanosleep(CLOCK_MONOTONIC, TIMER_ABSTIME, &ts, NULL);

            if(discrete_state->ETHERCAT_CONFIGURED == 1)
            {
                if(discrete_state->ETHERCAT_RUNNING == 1)
                {
                    if(ec_receive_processdata(1000) > 0);
                    {
                        continuous_state->motor_x->read_output_from_ECAT_buffer();
                        continuous_state->motor_y->read_output_from_ECAT_buffer();
                        continuous_state->motor_z->read_output_from_ECAT_buffer();
                        continuous_state->encoder->read_output_from_ECAT_buffer();
                        continuous_state->limit_switch->read_output_from_ECAT_buffer();

                        pthread_mutex_lock(&((&ecx_context)->port->rx_mutex));
                        clock_gettime(CLOCK_MONOTONIC, continuous_state->RT_ts);
                        pthread_mutex_unlock(&((&ecx_context)->port->rx_mutex));
                        
                        discrete_state->ETHERCAT_NEW_DATA = 1;
                    }

                    (*continuous_state->execute_task_function)(this);

                    continuous_state->motor_x->write_input_to_ECAT_buffer();
                    continuous_state->motor_y->write_input_to_ECAT_buffer();
                    continuous_state->motor_z->write_input_to_ECAT_buffer();
                    continuous_state->encoder->write_input_to_ECAT_buffer();
                    continuous_state->limit_switch->write_input_to_ECAT_buffer();
                    ec_send_processdata();
                }

                if (ec_slave[0].hasdc)
                    {time_sync((int64_t) ec_DCtime, (int64_t) params->RT_thread_cycletime, (int64_t*) &time_offset);}
        
                average_sync_offset = abs(time_offset);
                for(int i = 0; i < sizeof(sync_offset_ar)/sizeof(sync_offset_ar[0])-1; i++)
                {   
                    sync_offset_ar[i] = sync_offset_ar[i+1];
                    average_sync_offset += abs(sync_offset_ar[i]);
                }
                sync_offset_ar[ (sizeof(sync_offset_ar)/sizeof(sync_offset_ar[0])) - 1] = time_offset;

                average_sync_offset = average_sync_offset/ (sizeof(sync_offset_ar)/sizeof(sync_offset_ar[0]));
                
                if(average_sync_offset < params->RT_sync_jitter)
                    {discrete_state->RT_SYNCED = 1;}
                else 
                    {discrete_state->RT_SYNCED = 0;}
            }
            else
                {usleep(2000);}
        }
    }

    void ECATActivity::start_RT_thread()
    {
        std::thread thread(&ECATActivity::RT_thread, this);
        thread.join();
    }

    void ECATActivity::activity_config()
    {
        remove_schedule_from_eventloop(&activity->schedule_table, (char*) "activity_config");
        switch (activity->lcsm.state)
        {
        case CREATION:
            add_schedule_to_eventloop(&activity->schedule_table, (char*) "creation");
            break;
        case RESOURCE_CONFIGURATION:
            add_schedule_to_eventloop(&activity->schedule_table, (char*) "resource_configuration");
            break;
        case CAPABILITY_CONFIGURATION:
            add_schedule_to_eventloop(&activity->schedule_table, (char*) "capability_configuration");
            break;
        case PAUSING:
            break;
        case RUNNING:
            add_schedule_to_eventloop(&activity->schedule_table, (char*) "running");
            break;
        case CLEANING:
            break;
        case DONE:
            break;
        }
    };

    /* ----- Creation ----- */
    void ECATActivity::creation_compute()
    {
        /* Allocate memory */
        continuous_state->motor_x      = new ecat::EL7047;
        continuous_state->motor_y      = new ecat::EL7047;
        continuous_state->motor_z      = new ecat::EposDrive;
        continuous_state->encoder      = new ecat::EL5102;
        continuous_state->limit_switch = new ecat::EL1002;
        
        /* EtherCAT flags */
        discrete_state->EXECUTE_RT_THREAD   = 1;
        discrete_state->ETHERCAT_NEW_DATA   = 0;
        discrete_state->ETHERCAT_CONFIGURED = 0;
        discrete_state->ETHERCAT_RUNNING    = 0;
        discrete_state->RT_SYNCED           = 0;

        /* Homing */
        discrete_state->HOMING_COMPLETE = 0;

        /* External signals */
        discrete_state->OPERATOR_COMPLETE      = 0;
        discrete_state->COMING_FROM_ADJUSTMENT = 0;

        /* Alignment */
        discrete_state->FLAG_ALIGNED = 0;

        /* FSM initial state */
        discrete_state->gantry_state = ECATActivity::GANTRY_HOMING;

        /* Continuous state — target positions */
        continuous_state->target_x = 0.0;
        continuous_state->target_y = 0.0;

        /* Velocity profile */
        continuous_state->v_ref_prev = 0.0;

        /* Camera / marker */
        continuous_state->marker_pixel_x      = 0.0;
        continuous_state->marker_pixel_y      = 0.0;
        continuous_state->marker_detected     = false;
        continuous_state->marker_prev_pixel_x = 0.0;
        continuous_state->marker_prev_pixel_y = 0.0;

        /* Steady state counter */
        continuous_state->steady_state_frame_count = 0;

        activity->state.lcsm_flags.creation_complete = true;
    }

    void ECATActivity::creation_coordinate()
    {
        if (coord_state->deinitialisation_request)
            activity->state.lcsm_protocol = DEINITIALISATION;

        if (activity->state.lcsm_flags.creation_complete)
        {
            activity->lcsm.state = RESOURCE_CONFIGURATION;
            update_super_state_lcsm_flags(&activity->state.lcsm_flags, activity->lcsm.state);
        }
    }

    void ECATActivity::creation_configure()
    {
        if (activity->lcsm.state != CREATION)
        {
            add_schedule_to_eventloop(&activity->schedule_table, (char*) "activity_config");
            remove_schedule_from_eventloop(&activity->schedule_table, (char*) "creation");
        }
    }

    void ECATActivity::creation()
    {
        ECATActivity::creation_compute();
        ECATActivity::creation_coordinate();
        ECATActivity::creation_configure();
    }

    /* ----- Resource configuration ---- */
    int ECATActivity::resource_configuration_compute()
    {
        activity->state.lcsm_flags.resource_configuration_failed = false;

        int config_result;
        switch (activity->state.lcsm_protocol){
        case INITIALISATION:
            int status;
            configure_from_file(params->configuration_file, &status);
            if(status != 1)
                {return 0;}

            continuous_state->motor_x->set_input_file(params->path_to_configuration_file_motor_x);
            continuous_state->motor_y->set_input_file(params->path_to_configuration_file_motor_y);
            continuous_state->motor_z->set_input_file(params->path_to_configuration_file_motor_z);
            continuous_state->encoder->set_input_file(params->path_to_configuration_file_encoder);

            continuous_state->motor_x->flags.KILLSWITCH_ENGAGED = &(continuous_state->motor_z->flags.FAULT_STATE);
            continuous_state->motor_y->flags.KILLSWITCH_ENGAGED = &(continuous_state->motor_z->flags.FAULT_STATE);
            continuous_state->motor_z->ECAT->flags.RT_SYNCED    = &(discrete_state->RT_SYNCED);
            
            if(init_ethercat(params->ifname) != 1) 
                {return 0;}
            if(continuous_state->motor_x->initialise(params->motor_x_number_in_chain) != 1)
                {return 0;}
            if(continuous_state->motor_y->initialise(params->motor_y_number_in_chain) != 1)
                {return 0;}
            if(continuous_state->motor_z->initialise() != 1)
                {return 0;}
            if(continuous_state->encoder->initialise() != 1)
                {return 0;}
            if(continuous_state->limit_switch->initialise() != 1)
                {return 0;}

            ecat_configure();
            
            discrete_state->ETHERCAT_CONFIGURED = 1;
            activity->state.lcsm_flags.resource_configuration_complete = true;
            break;

        case DEINITIALISATION:
            discrete_state->EXECUTE_RT_THREAD = 0;
            go_to_init_and_shutdown();
            printf("Ethercat shutdown\n");
            activity->state.lcsm_flags.resource_configuration_complete = true;
            break;
        }

        return 1;
    }

    void ECATActivity::resource_configuration_coordinate()
    {
        if (*coordination_state->deinitialisation_request)
            {activity->state.lcsm_protocol = DEINITIALISATION;}

        if (activity->state.lcsm_flags.resource_configuration_complete)
        {
            switch (activity->state.lcsm_protocol){
                case INITIALISATION:
                    activity->lcsm.state = CAPABILITY_CONFIGURATION;
                    break;
                case EXECUTION:
                    activity->lcsm.state = RUNNING;
                    break;
                case DEINITIALISATION:
                    activity->lcsm.state = DONE;
                    activity->state.lcsm_flags.deletion_complete = true;
                    break;
            }
        }
        else if(activity->state.lcsm_flags.resource_configuration_failed)
        {
            activity->lcsm.state = DONE;
            activity->state.lcsm_flags.deletion_complete = true;
        }

        update_super_state_lcsm_flags(&activity->state.lcsm_flags, activity->lcsm.state);
    }

    void ECATActivity::resource_configuration_configure()
    {
        if (activity->lcsm.state != RESOURCE_CONFIGURATION)
        {
            add_schedule_to_eventloop(&activity->schedule_table, (char*) "activity_config");
            remove_schedule_from_eventloop(&activity->schedule_table, (char*) "resource_configuration");
            activity->state.lcsm_flags.resource_configuration_complete = false;
        }
    }

    void ECATActivity::resource_configuration()
    {
        if(resource_configuration_compute() == 0)
            {activity->state.lcsm_flags.resource_configuration_failed = true;}
        else
            {activity->state.lcsm_flags.resource_configuration_failed = false;}
        
        resource_configuration_coordinate();
        resource_configuration_configure();
    }

    /* ----- Capability configuration ----- */
    void ECATActivity::capability_configuration_communicate()
    {}

    void ECATActivity::capability_configuration_compute()
    {
        if(continuous_state->motor_x->configure() != 1)
            {activity->state.lcsm_flags.capability_configuration_failed = true;}
        else if(continuous_state->motor_y->configure() != 1)
            {activity->state.lcsm_flags.capability_configuration_failed = true;}
        else if(continuous_state->motor_z->configure() != 1)
            {activity->state.lcsm_flags.capability_configuration_failed = true;}
        else if(continuous_state->encoder->configure() != 1)
            {activity->state.lcsm_flags.capability_configuration_failed = true;}
        else if(continuous_state->limit_switch->configure() != 1)
            {activity->state.lcsm_flags.capability_configuration_failed = true;}
        else
            {activity->state.lcsm_flags.capability_configuration_complete = true;}
    }

    void ECATActivity::capability_configuration_coordinate()
    {
        coordination_state_t* activity_coordination_state;
        activity_coordination_state = (coordination_state_t*) activity->state.coordination_state;
        if(*(activity_coordination_state->deinitialisation_request) == true)
            {activity->state.lcsm_protocol = DEINITIALISATION;}

        if(activity->state.lcsm_flags.capability_configuration_complete == true)
        {
            if(coordination_state->execution_request)
                {activity->state.lcsm_protocol = EXECUTION;}

            switch (activity->state.lcsm_protocol)
            {
                case EXECUTION:
                    discrete_state->ETHERCAT_RUNNING = 1;
                    activity->lcsm.state = RUNNING;
                    break;
                case DEINITIALISATION:
                    activity->lcsm.state = RESOURCE_CONFIGURATION;
                    break;
            }
            
            update_super_state_lcsm_flags(&activity->state.lcsm_flags, activity->lcsm.state);
        }
        else if(activity->state.lcsm_flags.capability_configuration_failed == true)
        {
            activity->lcsm.state = DONE;
            activity->state.lcsm_flags.deletion_complete = true;
            update_super_state_lcsm_flags(&activity->state.lcsm_flags, activity->lcsm.state);
        }
    }

    void ECATActivity::capability_configuration_configure()
    {
        if (activity->lcsm.state != CAPABILITY_CONFIGURATION)
        {
            add_schedule_to_eventloop(&activity->schedule_table, (char*) "activity_config");
            remove_schedule_from_eventloop(&activity->schedule_table, (char*) "capability_configuration");
            activity->state.lcsm_flags.capability_configuration_failed  = false;
            activity->state.lcsm_flags.capability_configuration_complete = false;
        }
    }

    void ECATActivity::capability_configuration()
    {
        capability_configuration_communicate();
        capability_configuration_compute();
        capability_configuration_coordinate();
        capability_configuration_configure();
    }

    /* ----- Running ----- */
    void ECATActivity::running_compute()
    {
        if(discrete_state->ETHERCAT_NEW_DATA == 1)
        {
            discrete_state->ETHERCAT_NEW_DATA = 0;
            (*continuous_state->execute_FSM)(this);
        }
    }

    void ECATActivity::running_coordinate()
    {
        if (*coordination_state->configuration_request){
            activity->lcsm.state = CAPABILITY_CONFIGURATION;
            *coordination_state->configuration_request = false;
        }
        if (*coordination_state->deinitialisation_request){
            activity->state.lcsm_protocol = DEINITIALISATION;
            activity->lcsm.state = RESOURCE_CONFIGURATION;
        }
        update_super_state_lcsm_flags(&activity->state.lcsm_flags, activity->lcsm.state);
    }

    void ECATActivity::running_configure()
    {
        if (activity->lcsm.state != RUNNING)
        {
            add_schedule_to_eventloop(&activity->schedule_table, (char*) "activity_config");
            remove_schedule_from_eventloop(&activity->schedule_table, (char*) "running");
        }
    }

    void ECATActivity::running()
    {
        running_compute();
        running_coordinate();
        running_configure();
    }

    /* ----- Scheduler ----- */
    void ECATActivity::register_schedules()
    {
        schedule_t schedule_config = {.number_of_functions = 0};
        register_function(&schedule_config,(function_ptr_t) &ECATActivity::activity_config, this, (char*) "activity_config");
        register_schedule(&activity->schedule_table, schedule_config, (char*) "activity_config");

        schedule_t schedule_creation = {.number_of_functions = 0};
        register_function(&schedule_creation,(function_ptr_t) &ECATActivity::creation, this, (char*) "creation");
        register_schedule(&activity->schedule_table, schedule_creation, (char*) "creation");

        schedule_t schedule_resource_configuration = {.number_of_functions = 0};
        register_function(&schedule_resource_configuration,(function_ptr_t) &ECATActivity::resource_configuration, this, (char*) "resource_configuration");
        register_schedule(&activity->schedule_table, schedule_resource_configuration, (char*) "resource_configuration");

        schedule_t schedule_capability_configuration = {.number_of_functions = 0};
        register_function(&schedule_capability_configuration,(function_ptr_t) &ECATActivity::capability_configuration, this, (char*) "capability_configuration");
        register_schedule(&activity->schedule_table, schedule_capability_configuration, (char*) "capability_configuration");

        schedule_t schedule_running = {.number_of_functions = 0};
        register_function(&schedule_running,(function_ptr_t) &ECATActivity::running, this, (char*) "running");
        register_schedule(&activity->schedule_table, schedule_running, (char*) "running");
    }

    /* ----- (De)constructor ----- */
    ECATActivity::ECATActivity()
    {
        activity = new activity_t;

        create_lcsm();
        resource_configure_lcsm();

        discrete_state->EXECUTE_RT_THREAD = 1;
    }

    ECATActivity::~ECATActivity()
    {
        destroy_lcsm();
        delete activity;
    }

    /* ----- Activity LCSM ----- */
    void ECATActivity::create_lcsm()
    {
        params             = new ECATActivity::params_t;
        discrete_state     = new ECATActivity::discrete_state_t;
        coordination_state = new ECATActivity::coordination_state_t;
        continuous_state   = new ECATActivity::continuous_state_t;
        continuous_state->RT_ts = new struct timespec;
        
        continuous_state->execute_task_function = &ARNOLD::do_nothing;

        activity->conf.params = (params_t*) params;
        activity->state.computational_state.continuous = (continuous_state_t*) continuous_state;
        activity->state.computational_state.discrete   = (discrete_state_t*) discrete_state;
        activity->state.coordination_state             = (coordination_state_t*) coordination_state;
    }

    void ECATActivity::resource_configure_lcsm()
    {
        configure_lcsm_activity(activity, activity->table);
        activity->lcsm.state        = CREATION;
        activity->state.lcsm_protocol = INITIALISATION;

        ECATActivity::register_schedules();
        add_schedule_to_eventloop(&activity->schedule_table, (char*) "activity_config");
    }

    void ECATActivity::destroy_lcsm()
        {destroy_activity(activity);}

    void ECATActivity::configure_from_file(const char *file_path, int *status)
    {
        int number_of_params = 0;
        param_array_t param_array[32];

        /* Hardware configuration */
        param_array[number_of_params++] = (param_array_t){"path_to_configuration_file_motor_x", 
            &(params->path_to_configuration_file_motor_x), PARAM_TYPE_CHAR};
        param_array[number_of_params++] = (param_array_t){"path_to_configuration_file_motor_y", 
            &(params->path_to_configuration_file_motor_y), PARAM_TYPE_CHAR};
        param_array[number_of_params++] = (param_array_t){"path_to_configuration_file_motor_z", 
            &(params->path_to_configuration_file_motor_z), PARAM_TYPE_CHAR};
        param_array[number_of_params++] = (param_array_t){"path_to_configuration_file_encoder", 
            &(params->path_to_configuration_file_encoder), PARAM_TYPE_CHAR};
        param_array[number_of_params++] = (param_array_t){"FSM_schedule", 
            &(params->FSM_schedule), PARAM_TYPE_CHAR};
        param_array[number_of_params++] = (param_array_t){"ifname", 
            &(params->ifname), PARAM_TYPE_CHAR};
        param_array[number_of_params++] = (param_array_t){"motor_x_number_in_chain", 
            &(params->motor_x_number_in_chain), PARAM_TYPE_INT};
        param_array[number_of_params++] = (param_array_t){"motor_y_number_in_chain", 
            &(params->motor_y_number_in_chain), PARAM_TYPE_INT};
        param_array[number_of_params++] = (param_array_t){"RT_thread_cycletime", 
            &(params->RT_thread_cycletime), PARAM_TYPE_INT};
        param_array[number_of_params++] = (param_array_t){"RT_sync_jitter", 
            &(params->RT_sync_jitter), PARAM_TYPE_INT};

        /* Steady state detection (slide 5) */
        param_array[number_of_params++] = (param_array_t){"steady_state_threshold_px",
            &(params->steady_state_threshold_px), PARAM_TYPE_DOUBLE};
        param_array[number_of_params++] = (param_array_t){"steady_state_n_frames",
            &(params->steady_state_n_frames), PARAM_TYPE_INT};

        /* Velocity profile + position feedback (slides 3 & 4) */
        param_array[number_of_params++] = (param_array_t){"L_cable",
            &(params->L_cable), PARAM_TYPE_DOUBLE};
        param_array[number_of_params++] = (param_array_t){"v_max",
            &(params->v_max), PARAM_TYPE_DOUBLE};
        param_array[number_of_params++] = (param_array_t){"kp_position",
            &(params->kp_position), PARAM_TYPE_DOUBLE};
        param_array[number_of_params++] = (param_array_t){"dt",
            &(params->dt), PARAM_TYPE_DOUBLE};
        param_array[number_of_params++] = (param_array_t){"epsilon_x",
            &(params->epsilon_x), PARAM_TYPE_DOUBLE};
        param_array[number_of_params++] = (param_array_t){"epsilon_v",
            &(params->epsilon_v), PARAM_TYPE_DOUBLE};

        /* Target correction (slide 7) */
        param_array[number_of_params++] = (param_array_t){"correction_step_mm",
            &(params->correction_step_mm), PARAM_TYPE_DOUBLE};

        int config_status_activity;
        read_from_input_file(file_path, param_array, number_of_params, &config_status_activity);

        if (config_status_activity == CONFIGURATION_FROM_FILE_SUCCEEDED)
            {*status = CONFIGURATION_FROM_FILE_SUCCEEDED;}
        else
            {*status = CONFIGURATION_FROM_FILE_FAILED;}

        /* Setup FSM — deux schedules possibles */
        if(strcmp(params->FSM_schedule, "homing") == 0)
            {continuous_state->execute_FSM = &ARNOLD::homing_FSM;}
        else if(strcmp(params->FSM_schedule, "gantry") == 0)
            {continuous_state->execute_FSM = &ARNOLD::gantry_FSM;}
        else
            {*status = CONFIGURATION_FROM_FILE_FAILED;}
    }
}