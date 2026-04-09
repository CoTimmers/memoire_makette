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

// #include "beckhoff_modules/ebox.cpp"
#include "beckhoff_modules/EL7047.hpp"
#include "beckhoff_modules/EL5102.hpp"

#include "time_functions/time_functions.h"
#include "SOEM_helper_functions/etherCAT_communication.h"

#include <iostream>
// #include <bitset>

using namespace std;

int run = 0;
int execute_thread = 1;

void signalHandler( int signum ) 
{
    cout << "Interrupt signal (" << signum << ") received.\n";
    
    /* Close down SOEM connection */
    execute_thread = 0;

    usleep(1000000);
    printf("Ending program\n");
    exit(signum);  
}

void print_IO(ecat::EL7047 *motor_x,ecat::EL7047 *motor_y, ecat::EL5102 *encoder)
{
    cout << "DRIVE X" << endl;
    // printf("enable %i\n", motor_x->get_enable());
    // printf("input %i\n", motor_x->ECAT->m_ECAT_inputs->input);
    // printf("ready_to_enable %i\n", motor_x->ECAT->m_ECAT_outputs->control_status.ready_to_enable);
    // printf("ready %i\n", motor_x->ECAT->m_ECAT_outputs->control_status.ready);

    cout << "DRIVE Y" << endl;
    // printf("enable %i\n", motor_y->get_enable());
    // printf("input %i\n", motor_y->ECAT->m_ECAT_inputs->input);
    // printf("ready_to_enable %i\n", motor_y->ECAT->m_ECAT_outputs->control_status.ready_to_enable);
    // printf("ready %i\n", motor_y->ECAT->m_ECAT_outputs->control_status.ready);
     
    cout << "ENCODER" << endl;
    // cout << "counter_value: " << encoder->ECAT->m_ECAT_outputs->channel1.counter_value <<endl;
    cout << "position 1: " << encoder->channel1->get_position() <<endl;
    cout << "position 2: " << encoder->channel2->get_position() <<endl;
    cout << endl;
}

// int get_left_limit_switch(ecat::EboxData* ebox)
// {
//     int out = ebox->get_digital_output();
//     cout << "outl " << out << endl;
//     if((out & 0x02) == 0x02)
//         {return 0;}
//     return 1;
// }
// int get_right_limit_switch(ecat::EboxData* ebox)
// {
//     int out = ebox->get_digital_output();
//     cout << "outr " << out << endl;
//     if((out & 0x04) == 0x04)
//         {return 0;}
//     return 1;
// }

int start_homing(char* ifname, ecat::EL7047* drive,ecat::EL7047* drive2, ecat::EL5102* encoder)
// int start_homing(char* ifname, ecat::EL7047* drive, ecat::EL5102* encoder, ecat::EboxData* ebox)
{
    /* initialise SOEM, bind socket to ifname */
    if(init_ethercat(ifname) != 0) 
    {
        if(drive->initialise(1) != 1)
            {return 0;}
        if(drive2->initialise(2) != 1)
            {return 0;}
        if(encoder->initialise() != 1)
            {return 0;}
        // if(ebox->initialise() != 1)
        //     {return 0;}

        ecat_configure();

        if(drive->configure() != 1)
            {return 0;}
        if(drive2->configure() != 1)
            {return 0;}
        if(encoder->configure() != 1)
            {return 0;}
        // if(ebox->configure() != 1)
        //     {return 0;}
        
        drive->set_control_mode(EL7047_VELOCITY_CONTROL);
        drive2->set_control_mode(EL7047_VELOCITY_CONTROL);

        struct timespec   ts;
        clock_gettime(CLOCK_MONOTONIC, &ts);
        int cycletime = 2000000;    /* cycletime in ns */

        ec_send_processdata();
        ec_receive_processdata(EC_TIMEOUTRET);

        int direction = 1;     // left
        int calibrated = 0;
        double velocity = 5;

        double left;
        // /* Asynchronous loop */
        while(1)
        // while(calibrated != 1)
        {
            /* calculate next cycle start */
            add_timespec(&ts, cycletime);
            clock_nanosleep(CLOCK_MONOTONIC, TIMER_ABSTIME, &ts, NULL);
            ec_send_processdata();
            ec_receive_processdata(EC_TIMEOUTRET);

            drive->read_output_from_ECAT_buffer();
            drive2->read_output_from_ECAT_buffer();
            encoder->read_output_from_ECAT_buffer();

            // if((get_left_limit_switch(ebox) == 1) && (direction == -1))    // Motor at left most position 
            // {
            //     cout << "left" << endl;
            //     // set reference pulse 
            //     int pulses_per_revolution = encoder->get_pulses_per_revolution();

            //     encoder->channel1->set_counter(1);
            //     encoder->channel1->set_counter_value(
            //         (pulses_per_revolution*10)/4.43);
            //     // get left position
            //     direction *= -1;    // Switch direction  

            //     drive->set_enable(0);
            //     drive->set_input(0);
            // }

            // else if((get_right_limit_switch(ebox) == 1) && (direction == 1))    // Motor at left most position 
            // else if((get_right_limit_switch(ebox) == 1) && (direction == 1))    // Motor at left most position 
            // {
            //     cout << "right" << endl;
            //     // set reference 
            //     calibrated = 1; // End condition

            //     drive->set_enable(0);
            //     drive->set_input(0);
            // }
            // else
            {
                cout << "else" << endl;
                encoder->channel1->set_counter(0);
                drive2->set_enable(1);
                drive2->set_input(velocity*direction);
                // drive->set_enable(1);
                // drive->set_input(velocity*direction);
            }

            print_IO(drive,drive2,encoder);

            drive->write_input_to_ECAT_buffer();
            drive2->write_input_to_ECAT_buffer();
            encoder->write_input_to_ECAT_buffer();
        }


    }
    else
    {
        printf("Could not init ethercat\n");  
    }

    printf("Close socket\n");
    ec_close();

    return 1;
}

int main()
{
    // printf("SOEM (Simple Open EtherCAT Master)\nE/BOX test\n");
    
    char *ifname;
    ifname = (char*) "enx3c2c30f0fc24";
    printf("Ethernet card = %s\n",ifname);

    ecat::EL7047 *drive = new ecat::EL7047;
    ecat::EL7047 *drive2 = new ecat::EL7047;
    ecat::EL5102 *encoder = new ecat::EL5102;

    drive->set_input_file("../configuration_files/EL7047_configuration_data.json");
    drive2->set_input_file("../configuration_files/EL7047_configuration_data.json");
    encoder->set_input_file("../configuration_files/EL5102_configuration_data.json");

    start_homing(ifname, drive, drive2, encoder);
    // start_homing(ifname, drive, encoder, ebox);
}


