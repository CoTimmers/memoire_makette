/* ----------------------------------------------------------------------------
 * Project Title,
 * ROB @ KU Leuven, Leuven, Belgium
 * Authors: 
 * See LICENSE for the license information
 * -------------------------------------------------------------------------- */

/**
 * @file image_processing_activity.c
 * @date Ocotber 12, 2021
 **/

#include "string.h"
#include <time.h>
#include <unistd.h>
#include <iostream>

#include <opencv2/opencv.hpp>
#include "image_processing_activity/image_processing_activity.hpp"
#include "image_processing_activity/image_processing_functions.hpp"
// #include "image_processing_activity/basler_functions.hpp"
#include <five_c/cpp_baseclass/cpp_baseclass.hpp>
#include <read_file/read_file.h>

namespace activity_5c
{
    /* ----- Creation ----- */
    void ImageProcessingActivity::creation_compute()
    {
        mtx_init(&coordination_state->mutex, mtx_plain);
        activity->state.lcsm_flags.creation_complete = true;

        /* Assign name to schedules */
        strcpy(continuous_state->ip_running_schedule.name, (char*) "running_hook");
        strcpy(continuous_state->ip_configure_schedule.name, (char*) "configure_hook");
    }
// 
    /* ----- Resource configuration ---- */
    // void ImageProcessingActivity::resource_configuration_compute()
    // {

    // }
    /* ----- Capability Configuration ----- */
    void ImageProcessingActivity::capability_configuration_compute()
    {
        int configuration_failed = 0;

        hash_result_t hash_result;
        variable_registration_args_t registration_args;
        mtx_t *mutex;

        strcpy(registration_args.data.model, "basler_frame");    
        strcpy(registration_args.data.data_type, "cv::Mat");    

        variable_entry_pointer_t entry;
            GET_VARIABLE_IN_TABLE_WITHOUT_DATATYPE(activity->table, registration_args, 
               entry, hash_result);
        
        /* Wait for camera to make memory */
        if(hash_result == HASH_ENTRY_FOUND)
        {
            /* Insert new parameters of configuration file in hash table */ 
            if(read_and_add_parameters_to_table() != 1)
            {
                printf("ERROR: Could not read all parameters in configuration file '%s'\n", params->configuration_file);
                configuration_failed = 1;
            }
            /* Register functions of configuration file in configure schedule */
            else if(read_and_add_functions_to_schedule(&continuous_state->ip_configure_schedule, continuous_state->configure_function_arguments) != 1)
            {
                printf("ERROR: Could not construct all configure functions in configuration file '%s'\n", params->configuration_file);
                configuration_failed = 1;
            }
            /* Register functions of configuration file in running schedule */
            else if(read_and_add_functions_to_schedule(&continuous_state->ip_running_schedule, continuous_state->running_function_arguments) != 1)
            {
                printf("ERROR: Could not construct all running functions in configuration file '%s'\n", params->configuration_file);
                configuration_failed = 1;
            }
            /* Execute all the configuration functions */
            if(execute_schedule(&continuous_state->ip_configure_schedule) != 1)
            {
                configuration_failed = 1;
                printf("ERROR: Could not execute all configuration functions in configuration file '%s'\n", params->configuration_file);
            }

            if(configuration_failed != 1)
                {activity->state.lcsm_flags.capability_configuration_complete = true;}
        
        }
    }

    /* ---- Running ----- */
    void ImageProcessingActivity::running_compute()
    {
        if(coordination_state->captured_frame_toggle != NULL)  // check if pointer was linked
        {
            if(*coordination_state->captured_frame_toggle != coordination_state->captured_frame_toggle_old)
            {
                coordination_state->captured_frame_toggle_old = *coordination_state->captured_frame_toggle;
                /* Run function schedule */
                execute_schedule(&continuous_state->ip_running_schedule);
            }
        }
    }

    /* ----- Pausing ----- */

    /* ----- Cleaning ----- */
    void ImageProcessingActivity::cleaning_compute()
    {
        // delete ip_running_schedule;

        // activity->state.lcsm_flags.deletion_complete = true;
    }

    /* ----- (De)constructor ----- */
    ImageProcessingActivity::ImageProcessingActivity()
    {
        create_lcsm();
        resource_configure_lcsm();
    }

    ImageProcessingActivity::ImageProcessingActivity(variable_table_t *table)
    {
        variable_table = table;
                
        create_lcsm();
        resource_configure_lcsm();
    }

    ImageProcessingActivity::~ImageProcessingActivity()
        {}

    /* ----- Activity LCSM ----- */
    void ImageProcessingActivity::create_lcsm()
    {
        /* Remove previous allocated memory */
        delete (Activity::coordination_state_t*) Activity::coordination_state;

        /* Assign new memory */
        params = new params_t;
        continuous_state = new continuous_state_t;
        discrete_state = new discrete_state_t;
        coordination_state = new coordination_state_t;

        /* Link base class */
        Activity::coordination_state = coordination_state; 

        /* Link to activity struct if it is used outside of this */
        activity->conf.params = (params_t*) params;
        activity->state.computational_state.continuous = (continuous_state_t*) continuous_state;
        activity->state.computational_state.discrete = (discrete_state_t*) discrete_state;
        activity->state.coordination_state = (coordination_state_t*) coordination_state;
    }

    void ImageProcessingActivity::resource_configure_lcsm()
    {
        // Select the inital state of LCSM for this activity
        activity->lcsm.state = CREATION;
        activity->state.lcsm_protocol = INITIALISATION;

        configure_lcsm_activity(activity, variable_table);

        // Schedule table (adding config() for the first eventloop iteration)
        register_schedules();
        add_schedule_to_eventloop(&activity->schedule_table, (char*) "activity_config");
    }

    int ImageProcessingActivity::read_and_add_parameters_to_table()
    {
        /* Read parameters sequentionally from configuration, until there are no more */
        int i = 0;
        int read_status;
        do{
            /* Read name and type of parameter */
            std::string parameter_path = "parameters/parameter_" + std::to_string(i+1) + "/";
            std::string parameter_name_read = parameter_path + "name";
            std::string parameter_type_read = parameter_path + "type";
            
            char parameter_name[64];
            char parameter_type[64];

            param_array_t param_array[2]; 
            param_array[0] = (param_array_t){(char*) parameter_name_read.c_str(),&parameter_name, PARAM_TYPE_CHAR,OPTIONAL_PARAMETER};
            param_array[1] = (param_array_t){(char*) parameter_type_read.c_str(),&parameter_type, PARAM_TYPE_CHAR,OPTIONAL_PARAMETER};
            read_from_input_file(params->configuration_file, param_array, 2, &read_status);

            if (read_status == CONFIGURATION_FROM_FILE_FAILED)  // Something went wrong 
                {return 0;}
            else if(read_status == CONFIGURATION_FROM_FILE_SUCCEEDED)
            {
                /* Register in hash table */ 
                if(insert_variable_in_table(parameter_path, parameter_name, parameter_type) != 1)
                    {return 0;}
            }

            i++;    //Prepare for next iteration
        }while(read_status == CONFIGURATION_FROM_FILE_SUCCEEDED);

        return 1;
    }

    int ImageProcessingActivity::read_and_add_functions_to_schedule(sequential_schedule_t* schedule, ip_function_t function_arguments[MAX_IP_FUNCTIONS])
    {
        schedule->number_of_functions = 0    ;      // Reset

        /* Assign functions of running hook in schedule */
        int i = 0;
        int read_status;
        do{
            std::string schedule_name = schedule->name;
            std::string function_path = schedule_name + "/function_" + std::to_string(i+1) + "/";
            std::string function_name_read = function_path + "name";
            
            char function_name[64];

            param_array_t param_array[1];
            param_array[0] = (param_array_t){(char*) function_name_read.c_str(),&function_name, PARAM_TYPE_CHAR, OPTIONAL_PARAMETER};
            
            read_from_input_file(params->configuration_file, param_array, 1, &read_status);
            if (read_status == CONFIGURATION_FROM_FILE_FAILED)  // Something went wrong 
                {return 0;}
            else if(read_status == CONFIGURATION_FROM_FILE_SUCCEEDED)   
            {
                
                /* Construct function arguments */
                if(construct_function_arguments(function_path, function_name,  &function_arguments[i]) != 1)
                    {return 0;}
    
                /* Register functions in schedule */
                if(schedule->number_of_functions < MAX_IP_FUNCTIONS)
                {
                    // std::string function_name_in_schedule = "image_processing_function_"+ std::to_string(i); 
                    std::string function_name_in_schedule = (std::string) function_name + "_IP_function_" + std::to_string(i); 

                    register_function(schedule, &ImageProcessingActivity::execute_function, 
                    // register_function1(schedule, &ImageProcessingActivity::execute_function, 
                        (void*) &function_arguments[i],(char*) function_name_in_schedule.c_str());
                }
                else
                {
                    printf("ERROR: Trying to write more funcions than allowed in image processing schedule!\n");
                    return 0;
                }
            }
            
            i++;    //Prepare for next iteration
        }while(read_status == CONFIGURATION_FROM_FILE_SUCCEEDED);

        return 1;
    }

}
