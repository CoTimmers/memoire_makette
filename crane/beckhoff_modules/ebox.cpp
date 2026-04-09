#include <stdio.h>
#include <string.h>
#include <inttypes.h>
#include <unistd.h>
#include <stdlib.h>
#include <iostream>

#include "ethercat.h"
#include "ebox.hpp"

#include "SOEM_helper_functions/etherCAT_communication.h"

using namespace std;

namespace ecat
{

    int EboxData::initialise()
    {
        /* Link correct follower id */
        if(write_follower_number(m_follower_name,  
            &m_follower_no) == 0)
            {return 0;}

        return 1;
    }

    int EboxData::configure()
    {
        if(go_to_SAFEOP(m_follower_no) == 0)
            {return 0;}

        /* connect struct pointers to slave I/O pointers, SOEM defines I/O from follower perspective */
        assign_IO_with_SOEM();
        *m_inputs->digital_in = 0x00;

        if(go_to_OP(m_follower_no, 1000000000) == 0)
            {return 0;}

        return 1;
    }

    /* <EboxData Class> */
    EboxData::EboxData()
    {
        m_inputs    = new struct EBOX_inputs_ptr_s;
        m_inputs->control = new uint8_t;
        m_inputs->digital_in = new uint8_t;
        m_inputs->analog_in[0] = new int16_t;
        m_inputs->analog_in[1] = new int16_t;
        m_inputs->pwm_in[0] = new uint16_t;
        m_inputs->pwm_in[1] = new uint16_t;

        m_outputs   = new struct EBOX_outputs_ptr_s;
        m_outputs->status = new uint8_t;
        m_outputs->counter = new uint8_t;
        m_outputs->digital_out = new uint8_t;
        m_outputs->analog_out[0] = new int32_t;
        m_outputs->analog_out[1] = new int32_t;
        m_outputs->tsain = new uint32_t;
        m_outputs->encoder[0] = new int32_t;
        m_outputs->encoder[1] = new int32_t;

        m_follower_name = "E/BOX";
    }
    EboxData::~EboxData()
    {
        delete m_inputs->control;
        delete m_inputs->digital_in;
        delete m_inputs->analog_in[0];
        delete m_inputs->analog_in[1];
        delete m_inputs->pwm_in[0];
        delete m_inputs->pwm_in[1];
        delete m_inputs;

        delete m_outputs->status;
        delete m_outputs->counter;
        delete m_outputs->digital_out;
        delete m_outputs->analog_out[0];
        delete m_outputs->analog_out[1];
        delete m_outputs->tsain;
        delete m_outputs->encoder[0];
        delete m_outputs->encoder[1];
        delete m_outputs;
    }

    /* Links inputs struct to SOEM etherCAT inputs */
    void EboxData::assign_IO_with_SOEM()
    {
        EBOX_inputs_t *EBOX_inputs;
        EBOX_outputs_t *EBOX_outputs;

        /* !EtherCAT defines inputs and outputs the otherway around! */
        EBOX_inputs = (EBOX_inputs_t*) ec_slave[m_follower_no].outputs;
        EBOX_outputs = (EBOX_outputs_t*) ec_slave[m_follower_no].inputs; 

        m_inputs->control = &(EBOX_inputs->control);
        m_inputs->digital_in = &(EBOX_inputs->digital_in);
        m_inputs->analog_in[0] = &(EBOX_inputs->analog_in[0]);
        m_inputs->analog_in[1] = &(EBOX_inputs->analog_in[1]);
        m_inputs->pwm_in[0] = &(EBOX_inputs->pwm_in[0]);
        m_inputs->pwm_in[1] = &(EBOX_inputs->pwm_in[1]);

        m_outputs->status = &(EBOX_outputs->status);
        m_outputs->counter = &(EBOX_outputs->counter);
        m_outputs->digital_out = &(EBOX_outputs->digital_out);
        m_outputs->analog_out[0] = &(EBOX_outputs->analog_out[0]);
        m_outputs->analog_out[1] = &(EBOX_outputs->analog_out[1]);
        m_outputs->tsain = &(EBOX_outputs->tsain);
        m_outputs->encoder[0] = &(EBOX_outputs->encoder[0]);
        m_outputs->encoder[1] = &(EBOX_outputs->encoder[1]);

        /* Reroute mutexes from SOEM */
        m_input_mutex = &((&ecx_context)->port->tx_mutex); 
        m_output_mutex = &((&ecx_context)->port->rx_mutex); 
    }

    /* RW */
    void EboxData::set_digital_input(uint8_t t_input)
    {
        pthread_mutex_lock(m_input_mutex);
        *m_inputs->digital_in = t_input;
        pthread_mutex_unlock(m_input_mutex);
    }
    const uint8_t EboxData::get_digital_input() const 
    {
        pthread_mutex_lock(m_input_mutex);
        int t_input = *m_inputs->digital_in;
        pthread_mutex_unlock(m_input_mutex);
        return t_input;
    }
    const uint8_t EboxData::get_digital_output() const 
    {
        pthread_mutex_lock(m_input_mutex);
        int t_input = *m_outputs->digital_out;
        pthread_mutex_unlock(m_input_mutex);
        return t_input;
    }
    const uint8_t* EboxData::get_digital_input_address() const 
        {return m_inputs->digital_in;}

    /* Functions */
    void EboxData::set_dc_time()
    {
         /* maximum data rate for E/BOX v1.0.1 is around 150kHz */
        int SYNC0TIME = 8000;
        ec_dcsync0(m_follower_no, TRUE, SYNC0TIME, 0); // SYNC0 on slave 1
    }

    /* <\EboxData Class> */

} // namespace ecat

