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

#include "image_processing_activity/image_processing_activity.hpp"
#include "image_processing_activity/image_processing_functions.hpp"
#include <five_c/cpp_baseclass/cpp_baseclass.hpp>
#include <read_file/read_file.h>

namespace activity_5c
{   
    int ImageProcessingActivity::insert_variable_in_table(std::string parameter_path, char* parameter_name, char* parameter_type)
    {
        std::string construction_path = parameter_path + "construction/";

        /* Initialise variable to get pointer to be assigned in hash table */
        hash_result_t hash_result;
        variable_registration_args_t registration_args;
        int data_size;

        strcpy(registration_args.data.model, parameter_name);        
        strcpy(registration_args.data.data_type, parameter_type);        

        int i  = 0;
        param_array_t param_array[16]; 

        int const_var_int[16];
        double const_var_double[16]{0}; // Initialise array with zeros
        char const_var_char[256]{};  // Initialise empty string

        std::string value_read[16]; 

        // printf("Parameter_name: %s, parameter_type: %s\n", parameter_name, parameter_type); // uncomment for debugging 

        /* Check different defined types */
        if(strcmp(parameter_type, "cv::Rect") == 0)
        {
            data_size = sizeof(cv::Rect);
            
            value_read[0] = (construction_path+"x");
            value_read[1] = (construction_path+"y");
            value_read[2] = (construction_path+"width");
            value_read[3] = (construction_path+"height");

            param_array[i] = (param_array_t){value_read[i].c_str(),&const_var_int[i], PARAM_TYPE_INT};i++;
            param_array[i] = (param_array_t){value_read[i].c_str(),&const_var_int[i], PARAM_TYPE_INT};i++;
            param_array[i] = (param_array_t){value_read[i].c_str(),&const_var_int[i], PARAM_TYPE_INT};i++;
            param_array[i] = (param_array_t){value_read[i].c_str(),&const_var_int[i], PARAM_TYPE_INT};i++;
        }
        else if(strcmp(parameter_type, "cv::Mat") == 0)
        {
            data_size = sizeof(cv::Mat);
            value_read[0] = (construction_path+"image_path");
                       
            param_array[i] = (param_array_t){value_read[i].c_str(),const_var_char, PARAM_TYPE_CHAR, OPTIONAL_PARAMETER};i++; // optional
        }
        else if(strcmp(parameter_type, "cv::Scalar") == 0)
        {
            data_size = sizeof(cv::Scalar);
            value_read[0] = (construction_path+"value1");
            value_read[1] = (construction_path+"value2");
            value_read[2] = (construction_path+"value3");
            value_read[3] = (construction_path+"value4");

            param_array[i] = (param_array_t){value_read[i].c_str(),&const_var_double[0], PARAM_TYPE_DOUBLE, OPTIONAL_PARAMETER};i++; // optional
            param_array[i] = (param_array_t){value_read[i].c_str(),&const_var_double[1], PARAM_TYPE_DOUBLE, OPTIONAL_PARAMETER};i++; // optional
            param_array[i] = (param_array_t){value_read[i].c_str(),&const_var_double[2], PARAM_TYPE_DOUBLE, OPTIONAL_PARAMETER};i++; // optional
            param_array[i] = (param_array_t){value_read[i].c_str(),&const_var_double[3], PARAM_TYPE_DOUBLE, OPTIONAL_PARAMETER};i++; // optional
        }
        else if(strcmp(parameter_type, "int") == 0)
        {
            data_size = sizeof(int);
            value_read[0] = construction_path+"value";
            
            param_array[i] = (param_array_t){(char*) (value_read[i].c_str()), &const_var_int[0], PARAM_TYPE_INT};i++; 
        }
        else if(strcmp(parameter_type, "double") == 0)
        {
            data_size = sizeof(double);
            value_read[0] = construction_path+"value";
                       
            param_array[i] = (param_array_t){value_read[i].c_str(), &const_var_double[0], PARAM_TYPE_DOUBLE};i++;
        }
        else if(strcmp(parameter_type, "char") == 0)
        {
            data_size = 256;
            value_read[0] = construction_path+"value";
                       
            param_array[i] = (param_array_t){value_read[0].c_str(), &const_var_char, PARAM_TYPE_CHAR};i++;
        }
        else
        {
            printf("WARNING: parameter '%s' of type '%s' is not defined (yet)!\n", parameter_name, parameter_type);
            return 0;
        }

        int read_status;
        read_from_input_file(params->configuration_file, param_array, i, &read_status);
        if (read_status == CONFIGURATION_FROM_FILE_FAILED)  // Something went wrong 
            {return 0;}

            variable_entry_pointer_t variable_entry_pointer;
        /* Register and assign variable in hash table */
        // if(strcmp(parameter_type, "cv::Mat") == 0)
        // {
        //     cv::Mat *mat;
        //     mtx_t *mutex;
        //     REGISTER_AND_GET_VARIABLE_IN_TABLE(activity->table, registration_args, cv::Mat, mat, mutex, hash_result); 
        //     // Check if value was added or already exists 
        //     if( (hash_result == HASH_ENTRY_ADDED) || (hash_result == HASH_ENTRY_ALREADY_EXISTS) ) 
        //     {
        //         // Assign value to variable
        //         /* If different consructor types, add here */
        //         mtx_lock(mutex);
        //         if(const_var_char[0])   // None-empty string
        //             {*mat = imread(const_var_char, cv::IMREAD_COLOR);}
        //         mtx_unlock(mutex);
        //     }
        //     else    // Value not added, and does not exist
        //         {return 0;}
        // }
        // else
        // {
            // REGISTER_AND_GET_VARIABLE_IN_TABLE_GENERIC(activity->table, 
            //     registration_args, data_size, variable_entry_pointer, hash_result);
        // }
        if(strcmp(parameter_type, "cv::Mat") == 0)
        {
            REGISTER_AND_GET_VARIABLE_IN_TABLE(activity->table, registration_args, cv::Mat, 
                variable_entry_pointer.pointer, variable_entry_pointer.mutex, hash_result);           
        }
        else
        {
            REGISTER_AND_GET_VARIABLE_IN_TABLE_GENERIC(activity->table, 
                registration_args, data_size, variable_entry_pointer, hash_result);
        }

        // Check if value was added or already exists 
        if ((hash_result == HASH_ENTRY_ADDED) || (hash_result == HASH_ENTRY_ALREADY_EXISTS)) 
        {
            // Assign value to variable
            mtx_lock(variable_entry_pointer.mutex);
            if(strcmp(parameter_type, "cv::Rect") == 0)
            {
                cv::Rect* rect = (cv::Rect*) variable_entry_pointer.pointer;
                *rect = cv::Rect(const_var_int[0],const_var_int[1],const_var_int[2],const_var_int[3]);         
            }
            else if(strcmp(parameter_type, "cv::Mat") == 0)
            {
                cv::Mat *mat = (cv::Mat*) variable_entry_pointer.pointer;
                if(const_var_char[0] != NULL)   // None-empty string
                {
                    new (mat) cv::Mat();
                    *mat = imread(const_var_char, cv::IMREAD_COLOR);
                }
                else
                    {*(cv::Mat*) variable_entry_pointer.pointer = cv::Mat::zeros(1,1,CV_8UC3);}
            }
            else if(strcmp(parameter_type, "cv::Scalar") == 0)
            {
                cv::Scalar *scalar = (cv::Scalar*) variable_entry_pointer.pointer;
                *scalar = cv::Scalar(const_var_double[0],const_var_double[1],const_var_double[2],const_var_double[3]);         
            }
            else if(strcmp(parameter_type, "int") == 0)
            {
                int *value = (int*) variable_entry_pointer.pointer;
                *value = const_var_int[0];
            }
            else if(strcmp(parameter_type, "double") == 0)
            {
                
                double *value = (double*) variable_entry_pointer.pointer;
                *value = const_var_double[0];
            }
            else if(strcmp(parameter_type, "char") == 0)
            {
                strcpy((char*) variable_entry_pointer.pointer,const_var_char);
            }
            mtx_unlock(variable_entry_pointer.mutex);
        }
        else    // Value not added, and does not exist
            {return 0;}

        return 1;
    }


    /* TODO automate how outputs are added */
    int ImageProcessingActivity::construct_function_arguments(std::string function_path, char* function_name, ip_function_t *ip_function)
    {
        ip_function->number_of_input_arguments = 0;       // Reset
        ip_function->number_of_output_arguments = 0;       // Reset
        strcpy(ip_function->name, function_name);   // Name of function
        
        hash_result_t hash_result;
        variable_registration_args_t registration_args;

        char input_data_name[8][256];
        char output_data_name[8][256];

        char input_data_type[8][32];
        char output_data_type[8][32];

        char input_data_value[8][64];   
        char output_data_value[8][64]; 

        std::string input_function_var[8];
        std::string output_function_var[8];

        param_array_t param_array[16]; 

        /* Find relevant function */
        /* Read function arguments from configuration file */
        if(strcmp(ip_function->name, "crop_roi") == 0)
        {
            ip_function->number_of_input_arguments = 2;
            ip_function->number_of_output_arguments = 1;

            strcpy(input_data_name[0],(function_path+"inputs/input_frame").c_str());
            strcpy(input_data_name[1],(function_path+"inputs/crop_window").c_str());

            strcpy(output_data_name[0], (function_path+"outputs/output_frame").c_str());

            /* Data types of arguments */
            strcpy(input_data_type[0], (char*) "cv::Mat");
            strcpy(input_data_type[1], (char*) "cv::Rect");

            strcpy(output_data_type[0], (char*) "cv::Mat");
        }
        else if(strcmp(ip_function->name, "copy_image") == 0)
        {
            ip_function->number_of_input_arguments = 1;
            ip_function->number_of_output_arguments = 1;

            strcpy(input_data_name[0],(function_path+"inputs/input_frame").c_str());
            strcpy(output_data_name[0], (function_path+"outputs/output_frame").c_str());

            /* Data types of arguments */
            strcpy(input_data_type[0], (char*) "cv::Mat");
            strcpy(output_data_type[0], (char*) "cv::Mat");
        }
        else if(strcmp(ip_function->name, "display_frame") == 0)
        {
            ip_function->number_of_input_arguments = 3;
            ip_function->number_of_output_arguments = 0;

            strcpy(input_data_name[0],(function_path+"inputs/window_name").c_str());
            strcpy(input_data_name[1],(function_path+"inputs/frame").c_str());
            strcpy(input_data_name[2],(function_path+"inputs/time").c_str());

            /* Data types of arguments */
            strcpy(input_data_type[0], (char*) "char");
            strcpy(input_data_type[1], (char*) "cv::Mat");
            strcpy(input_data_type[2], (char*) "int");
        }
        else if(strcmp(ip_function->name, "colour_detection_hsv") == 0)
        {
            ip_function->number_of_input_arguments = 3;
            ip_function->number_of_output_arguments = 1;

            strcpy(input_data_name[0],(function_path+"inputs/input_frame").c_str());
            strcpy(input_data_name[1],(function_path+"inputs/lower_colour_threshold").c_str());
            strcpy(input_data_name[2],(function_path+"inputs/upper_colour_threshold").c_str());
            
            strcpy(output_data_name[0], (function_path+"outputs/output_frame").c_str());

            /* Data types of arguments */
            strcpy(input_data_type[0], (char*) "cv::Mat");
            strcpy(input_data_type[1], (char*) "cv::Scalar");
            strcpy(input_data_type[2], (char*) "cv::Scalar");

            strcpy(output_data_type[0], (char*) "cv::Mat");
        }
        else if(strcmp(ip_function->name, "median_blur_filter") == 0)
        {
            ip_function->number_of_input_arguments = 2;
            ip_function->number_of_output_arguments = 1;

            strcpy(input_data_name[0],(function_path+"inputs/input_frame").c_str());
            strcpy(input_data_name[1],(function_path+"inputs/filter_size").c_str());
            
            strcpy(output_data_name[0], (function_path+"outputs/output_frame").c_str());

            /* Data types of arguments */
            strcpy(input_data_type[0], (char*) "cv::Mat");
            strcpy(input_data_type[1], (char*) "int");

            strcpy(output_data_type[0], (char*) "cv::Mat");
        }
        else if(strcmp(ip_function->name, "calculate_centre_point_and_area") == 0)
        {
            ip_function->number_of_input_arguments = 2;
            ip_function->number_of_output_arguments = 2;

            strcpy(input_data_name[0],(function_path+"inputs/input_frame").c_str());
            strcpy(input_data_name[1],(function_path+"inputs/minimum_area").c_str());

            strcpy(output_data_name[0], (function_path+"outputs/point").c_str());
            strcpy(output_data_name[1], (function_path+"outputs/area").c_str());

            /* Read function arguments from configuration file */
            strcpy(input_data_type[0], (char*) "cv::Mat");
            strcpy(input_data_type[1], (char*) "double");

            strcpy(output_data_type[0], (char*) "cv::Point");
            strcpy(output_data_type[1], (char*) "double");
        }
        else if(strcmp(ip_function->name, "transform_point_in_rect") == 0)
        {
            ip_function->number_of_input_arguments = 2;
            ip_function->number_of_output_arguments = 1;

            strcpy(input_data_name[0],(function_path+"inputs/input_point").c_str());
            strcpy(input_data_name[1],(function_path+"inputs/transform_window").c_str());
            
            strcpy(output_data_name[0], (function_path+"outputs/output_point").c_str());
            
            /* Read function arguments from configuration file */
            strcpy(input_data_type[0], (char*) "cv::Point");
            strcpy(input_data_type[1], (char*) "cv::Rect");

            strcpy(output_data_type[0], (char*) "cv::Point");
        }
        else if(strcmp(ip_function->name, "draw_circle_on_frame") == 0)
        {
            ip_function->number_of_input_arguments = 4;
            ip_function->number_of_output_arguments = 0;

            strcpy(input_data_name[0],(function_path+"inputs/input_frame").c_str());
            strcpy(input_data_name[1],(function_path+"inputs/centre").c_str());
            strcpy(input_data_name[2],(function_path+"inputs/radius").c_str());
            strcpy(input_data_name[3],(function_path+"inputs/colour").c_str());

            /* Data types of arguments */
            strcpy(input_data_type[0], (char*) "cv::Mat");
            strcpy(input_data_type[1], (char*) "cv::Point");
            strcpy(input_data_type[2], (char*) "int");
            strcpy(input_data_type[3], (char*) "cv::Scalar");
        }
        else if(strcmp(ip_function->name, "draw_rectangle_on_frame") == 0)
        {
            ip_function->number_of_input_arguments = 4;
            ip_function->number_of_output_arguments = 0;

            strcpy(input_data_name[0],(function_path+"inputs/input_frame").c_str());
            strcpy(input_data_name[1],(function_path+"inputs/rectangle").c_str());
            strcpy(input_data_name[2],(function_path+"inputs/colour").c_str());
            strcpy(input_data_name[3],(function_path+"inputs/thickness").c_str());

            /* Read function arguments from configuration file */
            strcpy(input_data_type[0], (char*) "cv::Mat");
            strcpy(input_data_type[1], (char*) "cv::Rect");
            strcpy(input_data_type[2], (char*) "cv::Scalar");
            strcpy(input_data_type[3], (char*) "int");
        }
        else if(strcmp(ip_function->name, "construct_rectangle_around_point") == 0)
        {
            ip_function->number_of_input_arguments = 2;
            ip_function->number_of_output_arguments = 1;

            strcpy(input_data_name[0],(function_path+"inputs/point").c_str());
            strcpy(input_data_name[1],(function_path+"inputs/distance_from_centre").c_str());

            strcpy(output_data_name[0], (function_path+"outputs/rectangle").c_str());
            
            /* Read function arguments from configuration file */
            strcpy(input_data_type[0], (char*) "cv::Point");
            strcpy(input_data_type[1], (char*) "int");
            
            strcpy(output_data_type[0], (char*) "cv::Rect");
        }
        else if(strcmp(ip_function->name, "fit_rectangle_in_frame") == 0)
        {
            ip_function->number_of_input_arguments = 2;
            ip_function->number_of_output_arguments = 1;

            strcpy(input_data_name[0],(function_path+"inputs/input_rectangle").c_str());
            strcpy(input_data_name[1],(function_path+"inputs/frame").c_str());

            strcpy(output_data_name[0], (function_path+"outputs/output_rectangle").c_str());

            /* Data types of arguments */
            strcpy(input_data_type[0], (char*) "cv::Rect");
            strcpy(input_data_type[1], (char*) "cv::Mat");

            strcpy(output_data_type[0], (char*) "cv::Rect");
        }
        else if(strcmp(ip_function->name, "read_undistortion_mappings") == 0)
        {
            ip_function->number_of_input_arguments = 1;
            ip_function->number_of_output_arguments = 2;

            strcpy(input_data_name[0],(function_path+"inputs/path_to_camera_parameters_file").c_str());
            
            strcpy(output_data_name[0], (function_path+"outputs/camera_matrix").c_str());
            strcpy(output_data_name[1], (function_path+"outputs/distortion_matrix").c_str());
            
            /* Data types of arguments */
            strcpy(input_data_type[0], (char*) "char");

            strcpy(output_data_type[0], (char*) "cv::Mat");
            strcpy(output_data_type[1], (char*) "cv::Mat");
        }
        else if(strcmp(ip_function->name, "undistort_point") == 0)
        {
            ip_function->number_of_input_arguments = 3;
            ip_function->number_of_output_arguments = 1;

            strcpy(input_data_name[0],(function_path+"inputs/input_point").c_str());
            strcpy(input_data_name[1],(function_path+"inputs/camera_matrix").c_str());
            strcpy(input_data_name[2],(function_path+"inputs/distortion_matrix").c_str());
            
            strcpy(output_data_name[0], (function_path+"outputs/output_point").c_str());

            /* Data types of arguments */
            strcpy(input_data_type[0], (char*) "cv::Point");
            strcpy(input_data_type[1], (char*) "cv::Mat");
            strcpy(input_data_type[2], (char*) "cv::Mat");

            strcpy(output_data_type[0], (char*) "cv::Point");
        }
        else if(strcmp(ip_function->name, "undistort_frame") == 0)
        {
            ip_function->number_of_input_arguments = 3;
            ip_function->number_of_output_arguments = 1;

            strcpy(input_data_name[0],(function_path+"inputs/input_frame").c_str());
            strcpy(input_data_name[1],(function_path+"inputs/camera_matrix").c_str());
            strcpy(input_data_name[2],(function_path+"inputs/distortion_matrix").c_str());
            
            strcpy(output_data_name[0], (function_path+"outputs/output_frame").c_str());

            /* Data types of arguments */
            strcpy(input_data_type[0], (char*) "cv::Mat");
            strcpy(input_data_type[1], (char*) "cv::Mat");
            strcpy(input_data_type[2], (char*) "cv::Mat");

            strcpy(output_data_type[0], (char*) "cv::Mat");
        }
        else if(strcmp(ip_function->name, "get_apriltag_data") == 0)
        {
            ip_function->number_of_input_arguments = 1;
            ip_function->number_of_output_arguments = 1;

            strcpy(input_data_name[0],(function_path+"inputs/input_frame").c_str());

            strcpy(output_data_name[0],(function_path+"outputs/output_frame").c_str());

            /* Data types of arguments */
            strcpy(input_data_type[0], (char*) "cv::Mat");
            
            strcpy(output_data_type[0], (char*) "cv::Mat");
        }
        else if(strcmp(ip_function->name, "save_image_frame") == 0)
        {
            ip_function->number_of_input_arguments = 4;
            ip_function->number_of_output_arguments = 0;

            strcpy(input_data_name[0],(function_path+"inputs/frame").c_str());
            strcpy(input_data_name[1],(function_path+"inputs/file_path").c_str());
            strcpy(input_data_name[2],(function_path+"inputs/resize_factor").c_str());
            strcpy(input_data_name[3],(function_path+"inputs/image_quality").c_str());


            /* Data types of arguments */
            strcpy(input_data_type[0], (char*) "cv::Mat");
            strcpy(input_data_type[1], (char*) "char");
            strcpy(input_data_type[2], (char*) "double");
            strcpy(input_data_type[3], (char*) "int");
            
        }
        else
        {
            printf("ERROR: Invalid image processing function name: '%s'!\n", ip_function->name);
            return 0;
        }

        /* Load in inputs and outputs */
        for(int i = 0; i < ip_function->number_of_input_arguments; i++)
            {param_array[i] = (param_array_t){input_data_name[i], input_data_value[i], PARAM_TYPE_CHAR};}
        for(int i = 0; i < ip_function->number_of_output_arguments; i++)
            {param_array[ip_function->number_of_input_arguments+i] = (param_array_t){output_data_name[i], output_data_value[i], PARAM_TYPE_CHAR};}

        int read_status;
        read_from_input_file(params->configuration_file, param_array,
            ip_function->number_of_input_arguments + ip_function->number_of_output_arguments, &read_status);
        if (read_status != CONFIGURATION_FROM_FILE_SUCCEEDED)
        {
            printf("ERROR: Could not load argument(s) of function '%s'!\n", ip_function->name);
            return 0;
        }

        /* Make storage for function output, internal function checks whether it is already existent */
        for(int i = 0; i < ip_function->number_of_output_arguments; i++)
        {
            strcpy(registration_args.data.model, output_data_value[i]);        
            strcpy(registration_args.data.data_type, output_data_type[i]);        
        
            /* Check required datasize to allocate */
            int data_size;
            if(strcmp(output_data_type[i], "cv::Rect") == 0)
                {data_size = sizeof(cv::Rect);}
            else if(strcmp(output_data_type[i], "cv::Mat") == 0)
                {data_size = sizeof(cv::Mat);}  //useless now
            else if(strcmp(output_data_type[i], "cv::Scalar") == 0)
                {data_size = sizeof(cv::Scalar);}
            else if(strcmp(output_data_type[i], "cv::Point") == 0)
                {data_size = sizeof(cv::Point);}
            else if(strcmp(output_data_type[i], "int") == 0)
                {data_size = sizeof(int);}
            else if(strcmp(output_data_type[i], "double") == 0)
                {data_size = sizeof(double);}
            else if(strcmp(output_data_type[i], "char") == 0)
                {data_size = sizeof(char[256]);}
            else
            {
                printf("ERROR: type: '%s', is not yet implemented in construct_function_arguments()!", output_data_type[i]);
                return 0;    
            }

            if(strcmp(output_data_type[i], "cv::Mat") == 0)
            {
    REGISTER_AND_GET_VARIABLE_IN_TABLE(activity->table, registration_args, cv::Mat, 
        ip_function->output_arguments[i].pointer, ip_function->output_arguments[i].mutex, hash_result);           
            }
            else
            {
    /* Assign memory in hash table */
    REGISTER_AND_GET_VARIABLE_IN_TABLE_GENERIC(activity->table, 
        registration_args, data_size, ip_function->output_arguments[i], hash_result);
            }
            if( !((hash_result == HASH_ENTRY_ADDED) || (hash_result == HASH_ENTRY_ALREADY_EXISTS)) ) // Check if succesfully added 
            {
                printf("Could not find output argument: '%s' of function: '%s' in hash table, error %i\n", 
                    output_data_value[i], function_name, hash_result);
                return 0;
            }

            /* Give initial values */
            if(strcmp(output_data_type[i], "cv::Point") == 0)
            {
                (*(cv::Point*) ip_function->output_arguments[i].pointer).x = -9999;
                (*(cv::Point*) ip_function->output_arguments[i].pointer).y = -9999;
            }
            if(strcmp(output_data_type[i], "cv::Mat") == 0)
                {*(cv::Mat*) ip_function->output_arguments[i].pointer = cv::Mat::zeros(1,1,CV_8UC3);}

            /* Initialise value for specific outputs */
            /* Currently only for cv::Point */
            /* TODO add more data types */
        }

        /* Get variables for inputs of function */
        for(int i = 0; i < ip_function->number_of_input_arguments; i++)
        {
            strcpy(registration_args.data.model, input_data_value[i]);        
            strcpy(registration_args.data.data_type, input_data_type[i]);  

            GET_VARIABLE_IN_TABLE_WITHOUT_DATATYPE(activity->table, registration_args, 
                ip_function->input_arguments[i], hash_result);    
            if(hash_result != HASH_ENTRY_FOUND)
            {
                printf("Could not find input argument: '%s' of function: '%s' in hash table, error %i\n", 
                    input_data_value[i], function_name, hash_result);
                return 0;
            } 
        }

        return 1;
    }
   
    int ImageProcessingActivity::execute_function(void* in)
    {
        ip_function_t *ip_function = (ip_function_t*) in;
        char* function_name = ip_function->name;

        // printf("Function name: %s\n", function_name);

        /* Lock mutexes */
        // for(int i=0; i < ip_function->number_of_input_arguments;i++)     
        //     {mtx_lock((ip_function->input_arguments[i].mutex));
        //     // printf("input %s\n", ip_function->input_arguments[i].)
        //     } 
        // for(int i=0; i < ip_function->number_of_output_arguments;i++)
        //     {mtx_lock((ip_function->output_arguments[i].mutex));} 

        /* TODO: All of these should be made to pass void pointers */
        if(strcmp(function_name, "crop_roi") == 0)
        {
            return IP::crop_roi(*((cv::Mat*) ip_function->input_arguments[0].pointer), ((cv::Mat*) ip_function->output_arguments[0].pointer),
                *((cv::Rect*) ip_function->input_arguments[1].pointer));
        }
        else if(strcmp(function_name, "copy_image") == 0)
        {
            return IP::copy_image(*((cv::Mat*) ip_function->input_arguments[0].pointer), ((cv::Mat*) ip_function->output_arguments[0].pointer));
        }
        else if(strcmp(function_name, "display_frame") == 0)
        {
            return IP::display_frame(((char*) ip_function->input_arguments[0].pointer), ((cv::Mat*) ip_function->input_arguments[1].pointer),
                *((int*) ip_function->input_arguments[2].pointer));
        }
        else if(strcmp(function_name, "colour_detection_hsv") == 0)
        {
            return IP::colour_detection_hsv(*((cv::Mat*) ip_function->input_arguments[0].pointer), ((cv::Mat*) ip_function->output_arguments[0].pointer),
                *((cv::Scalar*) ip_function->input_arguments[1].pointer), *((cv::Scalar*) ip_function->input_arguments[2].pointer));
        }
        else if(strcmp(function_name, "median_blur_filter") == 0)
        {
            return IP::median_blur_filter(*((cv::Mat*) ip_function->input_arguments[0].pointer), 
                ((cv::Mat*) ip_function->output_arguments[0].pointer),*((int*) ip_function->input_arguments[1].pointer)); 
        }
        else if(strcmp(function_name, "calculate_centre_point_and_area") == 0)
        {
            return IP::calculate_centre_point_and_area(*((cv::Mat*) ip_function->input_arguments[0].pointer), ((cv::Point*) ip_function->output_arguments[0].pointer),
                ((double*) ip_function->output_arguments[1].pointer), *((double*) ip_function->input_arguments[1].pointer)); 
        }
        else if(strcmp(function_name, "transform_point_in_rect") == 0)
        {
            return IP::transform_point_in_rect(*((cv::Point*) ip_function->input_arguments[0].pointer), 
                ((cv::Point*) ip_function->output_arguments[0].pointer),*((cv::Rect*) ip_function->input_arguments[1].pointer)); 
        }
        else if(strcmp(function_name, "draw_circle_on_frame") == 0)
        {
            return IP::draw_circle_on_frame(*((cv::Mat*) ip_function->input_arguments[0].pointer), (*(cv::Point*) ip_function->input_arguments[1].pointer),
                *((int*) ip_function->input_arguments[2].pointer), *((cv::Scalar*) ip_function->input_arguments[3].pointer)); 
        }
        else if(strcmp(function_name, "draw_rectangle_on_frame") == 0)
        {
            return IP::draw_rectangle_on_frame(*((cv::Mat*) ip_function->input_arguments[0].pointer), (*(cv::Rect*) ip_function->input_arguments[1].pointer),
                *((cv::Scalar*) ip_function->input_arguments[2].pointer), *((int*) ip_function->input_arguments[3].pointer)); 
        }
        else if(strcmp(function_name, "construct_rectangle_around_point") == 0)
        {
            return IP::construct_rectangle_around_point(*((cv::Point*) ip_function->input_arguments[0].pointer), 
                ((cv::Rect*) ip_function->output_arguments[0].pointer), *((int*) ip_function->input_arguments[1].pointer)); 
        }
        else if(strcmp(function_name, "fit_rectangle_in_frame") == 0)
        {
            return IP::fit_rectangle_in_frame(*((cv::Rect*) ip_function->input_arguments[0].pointer), 
                ((cv::Rect*) ip_function->output_arguments[0].pointer), *((cv::Mat*) ip_function->input_arguments[1].pointer)); 
        }
        else if(strcmp(function_name, "read_undistortion_mappings") == 0)
        {
            return IP::read_undistortion_mappings( ((char*) ip_function->input_arguments[0].pointer), 
                ((cv::Mat*) ip_function->output_arguments[0].pointer), ((cv::Mat*) ip_function->output_arguments[1].pointer)); 
        }
        else if(strcmp(function_name, "undistort_point") == 0)
        {
            return IP::undistort_point( *((cv::Point*) ip_function->input_arguments[0].pointer), ((cv::Point*) ip_function->output_arguments[0].pointer),  
                *((cv::Mat*) ip_function->input_arguments[1].pointer), *((cv::Mat*) ip_function->input_arguments[2].pointer)); 
        }
        else if(strcmp(function_name, "undistort_frame") == 0)
        {
            return IP::undistort_frame( *((cv::Mat*) ip_function->input_arguments[0].pointer), ((cv::Mat*) ip_function->output_arguments[0].pointer),  
                *((cv::Mat*) ip_function->input_arguments[1].pointer), *((cv::Mat*) ip_function->input_arguments[2].pointer)); 
        }
        else if(strcmp(function_name, "get_apriltag_data") == 0)
        {
            return IP::get_apriltag_data( *((cv::Mat*) ip_function->input_arguments[0].pointer), ((cv::Mat*) ip_function->output_arguments[0].pointer)); 
        }
        else if(strcmp(function_name, "save_image_frame") == 0)
        {
            return IP::save_image_frame( *((cv::Mat*) ip_function->input_arguments[0].pointer), ((char*) ip_function->input_arguments[1].pointer), 
                *((double*) ip_function->input_arguments[2].pointer), *((int*) ip_function->input_arguments[3].pointer));
        }
        else 
            {printf("Warning: Invalid image processing function name '%s'\n", function_name);}
        
        /* Unlock mutexes */
        for(int i=0; i < ip_function->number_of_input_arguments;i++)
            {mtx_unlock((ip_function->input_arguments[i].mutex));} 
        for(int i=0; i < ip_function->number_of_output_arguments;i++)
            {mtx_unlock((ip_function->output_arguments[i].mutex));} 
    }
}


