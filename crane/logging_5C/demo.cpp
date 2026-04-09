/* ----------------------------------------------------------------------------
 * Project Title,
 * ROB @ KU Leuven, Leuven, Belgium
 * Authors: 
 * See LICENSE for the license information
 * -------------------------------------------------------------------------- */

/**
     * @file demo.c
 * @date Ocotber 23, 2021
 **/


#include <stdio.h>
#include <pthread.h>
#include <signal.h>
#include <string.h>
#include <math.h>

#include <five_c/activity/activity.h>
#include <five_c/thread/thread.h>
#include <coordination_libraries/lcsm/lcsm.h>

#include <logging_activity/logging_activity.hpp>

#include <pthread.h>
#include <unistd.h>

/**
 * This function is called when ctrl+c is pressed.
*/
bool deinitialisation_request = 0;
bool execution_request = 1;

int RUN = 1;

static void sigint_handler(int sig){
    printf("Deinitialising activity\n");
    deinitialisation_request = 1;
  
    RUN = 0;

    usleep(100000);
    printf("Shutting down\n");
}

 /* put some random stuff in the table for testing */
int *int_1, *int_2, *int_3;
float *float_1, *float_2; 
double *double_1, *double_2;
char* string_1;

double *double_runtime;
typedef struct  custom_s{
    uint32_t a1;
    uint32_t a2;
}custom_t;
custom_t *custom;

void insert_values_in_table(variable_table_t* table)
{
    hash_result_t hash_result;
    variable_registration_args_t registration_args;
    mtx_t* lock;

    strcpy(registration_args.data.model,"int_1");
    REGISTER_AND_GET_VARIABLE_IN_TABLE(table, registration_args, 
        int, int_1, lock, hash_result);
    strcpy(registration_args.data.model,"int_2");
    REGISTER_AND_GET_VARIABLE_IN_TABLE(table, registration_args,  
        int, int_2, lock, hash_result);
    strcpy(registration_args.data.model,"int_3");
    REGISTER_AND_GET_VARIABLE_IN_TABLE(table, registration_args, 
        int, int_3, lock, hash_result);
    strcpy(registration_args.data.model,"float_1");
    REGISTER_AND_GET_VARIABLE_IN_TABLE(table, registration_args, 
        float, float_1, lock, hash_result);
    strcpy(registration_args.data.model,"float_2");
    REGISTER_AND_GET_VARIABLE_IN_TABLE(table, registration_args, 
        float, float_2, lock, hash_result);
    strcpy(registration_args.data.model,"double_1");
    REGISTER_AND_GET_VARIABLE_IN_TABLE(table, registration_args, 
        double, double_1, lock, hash_result);
    strcpy(registration_args.data.model,"double_2");
    REGISTER_AND_GET_VARIABLE_IN_TABLE(table, registration_args, 
        double, double_2, lock, hash_result);
    strcpy(registration_args.data.model,"string_1");
    REGISTER_AND_GET_VARIABLE_IN_TABLE(table, registration_args, 
        char, string_1, lock, hash_result);

    *int_1 = 11;
    *int_2 = 12;
    *int_3 = 13;

    *float_1 = 11.5;
    *float_2 = 12.5;

    *double_1 = 13.15;
    *double_2 = -213.212;

    strcpy(string_1,"test");

    /* Custom type */
    strcpy(registration_args.data.model ,"custom");
    REGISTER_AND_GET_VARIABLE_IN_TABLE(table, registration_args, 
        custom_t, custom, lock, hash_result);

    custom->a1 = 0x080808;
    custom->a2 = 0x101010;

}

int main(int argc, char**argv){
    signal(SIGINT, sigint_handler);
    
    variable_table_t variable_table;
    create_variable_table(&variable_table);

    activity_5c::LoggingActivity logging_activity(&variable_table);         
    
    insert_values_in_table(&variable_table);
    
    logging_activity.coordination_state->deinitialisation_request = &deinitialisation_request;
    logging_activity.coordination_state->execution_request = &execution_request;
    // ### THREADS ### //
    thread_t thread_logging;

    // Create thread: data structure, thread name, cycle time in milliseconds 
    create_thread(&thread_logging, (char*) "logging", 50); // 50 ms

    // Register activities in threads
    register_activity(&thread_logging, logging_activity.activity, (char*) "logging_activity");

    // ### SHARED MEMORY ### //

    // Create POSIX threads   
    pthread_t pthread_logging;

    pthread_create( &pthread_logging, NULL, do_thread_loop, ((void*) &thread_logging));

    hash_result_t hash_result;
    variable_registration_args_t registration_args;
    mtx_t* lock;
        // printf("float %f\n");
    usleep(2000000);

    variable_table_t* variable_table_ptr = &variable_table;
    strcpy(registration_args.data.model,"double_runtime");
    REGISTER_AND_GET_VARIABLE_IN_TABLE(variable_table_ptr, registration_args, 
        double, double_runtime, lock, hash_result);
    *double_runtime = 0.23;
    while(deinitialisation_request == 0){
    }
    
    // Wait for threads to finish, which means all activities must properly finish and reach the dead LCSM state
    pthread_join(pthread_logging, NULL); 

    return 0;
}
