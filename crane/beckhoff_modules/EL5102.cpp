/* 
    Beckhoff EL5102, 2 channel encoder.

    Author: Boris Deroo 
    KU Leuven, ROB group 
*/

#include <stdio.h>
#include <string.h>
#include <inttypes.h>
#include <unistd.h>
#include <stdlib.h>
#include <iostream>
#include <math.h>       

#include "ethercat.h"
#include "beckhoff_modules/EL5102.hpp"

#include "SOEM_helper_functions/etherCAT_communication.h"
#include "SOEM_helper_functions/soem_mapping.hpp"
#include "read_file/read_file.h"

using namespace std;

namespace ecat
{
    EL5102::EL5102()
    {        
        channel1 = new IO_channel_t;
        channel1->_EL5102 = this;
        channel1->m_inputs = new struct IO_channel_t::inputs_s;
        channel1->m_outputs = new struct IO_channel_t::outputs_s;
        
        channel2 = new IO_channel_t;
        channel2->_EL5102 = this;
        channel2->m_inputs = new struct IO_channel_t::inputs_s;
        channel2->m_outputs = new struct IO_channel_t::outputs_s;
        
        m_input_mutex = new mtx_t;
        m_output_mutex = new mtx_t;

        mtx_init(m_input_mutex, mtx_plain );
        mtx_init(m_output_mutex, mtx_plain );

        ECAT = new ecat::ECAT_EL5102;
    }
    EL5102::~EL5102()
    {
        delete channel1;
        delete channel2;
        
        delete m_input_mutex;
        delete m_output_mutex;
        delete ECAT;
    }

    int EL5102::initialise()
    {
        /* Link correct follower id */
        if(write_follower_number(ECAT->get_follower_name(),  
            ECAT->get_follower_no_address()) == 0)
            {return 0;}

        /* Set TxPDO and RxPDO mapping */
        if(ECAT->configure_cyclic_PDO_mapping(m_input_file) != 1)
            {return 0;}

        return 1;
    }

    int EL5102::configure()
    {
        /* Config motors */
        if(config_motor() != 1)
            {return 0;}

        if(go_to_SAFEOP(ECAT->get_follower_no()) == 0)
            {return 0;}

        // if(PDO::PDO_set_and_check(ECAT->get_follower_no(), 0x8001,0x1a, FALSE,4,0xFFFB40,EC_TIMEOUTSAFE) != 1)
        //     {return 0;}

        /* connect struct pointers to slave I/O pointers, SOEM defines I/O from follower perspective */
        ECAT->assign_IO_with_SOEM();

        if(go_to_OP(ECAT->get_follower_no(), 1000000000) == 0)
            {return 0;}

        return 1;
    }



    /* ----- RW ----- */
    void EL5102::set_input_file(const char* t_input_file)
        {strcpy(m_input_file,t_input_file);}
 
    void EL5102::IO_channel_t::set_parent(EL5102 *t_EL5102){_EL5102 = t_EL5102;}

    double EL5102::IO_channel_t::get_position()
    {
        mtx_lock(_EL5102->m_output_mutex);
        const double t_position = m_outputs->position;
        mtx_unlock(_EL5102->m_output_mutex);
        return t_position;
    }
    double EL5102::IO_channel_t::get_velocity()
    {
        mtx_lock(_EL5102->m_output_mutex);
        const double t_position = m_outputs->velocity;
        mtx_unlock(_EL5102->m_output_mutex);
        return t_position;
    }
    void EL5102::IO_channel_t::set_counter(double t_counter)
    {
        mtx_lock(_EL5102->m_input_mutex);
        m_inputs->counter = t_counter;
        mtx_unlock(_EL5102->m_input_mutex);
    }
    void EL5102::IO_channel_t::set_counter_value(double t_counter_value)
    {
        mtx_lock(_EL5102->m_input_mutex);
        m_inputs->counter_value = t_counter_value+m_revolution_offset*m_pulses_per_revolution;
        mtx_unlock(_EL5102->m_input_mutex);
    }
    double EL5102::IO_channel_t::get_pulses_per_revolution()
    {
        mtx_lock(_EL5102->m_output_mutex);
        const double t_pulses_per_revolution = m_pulses_per_revolution;
        mtx_unlock(_EL5102->m_output_mutex);
        return t_pulses_per_revolution;
    }

    /* ----- I/O Adresses ----- */
    double* EL5102::IO_channel_t::get_position_address() const
        {return &(m_outputs->position);}

    double* EL5102::IO_channel_t::get_velocity_address() const
        {return &(m_outputs->velocity);}

    /* Configuration */
    int EL5102::config_motor()
    {
        /* Read from input file */
        int i = 0;
        param_array_t param_array[6];
        param_array[i] = (param_array_t) {"channel1/pulses_per_revolution", &channel1->m_pulses_per_revolution, PARAM_TYPE_DOUBLE}; i++;
        param_array[i] = (param_array_t) {"channel1/m_revolution_offset", &channel1->m_revolution_offset, PARAM_TYPE_DOUBLE}; i++;
        param_array[i] = (param_array_t) {"channel1/gear_ratio", &channel1->m_gear_ratio, PARAM_TYPE_DOUBLE}; i++;
        
        param_array[i] = (param_array_t) {"channel2/pulses_per_revolution", &channel2->m_pulses_per_revolution, PARAM_TYPE_DOUBLE}; i++;
        param_array[i] = (param_array_t) {"channel2/m_revolution_offset", &channel2->m_revolution_offset, PARAM_TYPE_DOUBLE}; i++;
        param_array[i] = (param_array_t) {"channel2/gear_ratio", &channel2->m_gear_ratio, PARAM_TYPE_DOUBLE}; i++;

        int status;
        read_from_input_file(m_input_file, param_array,i, &status);
        if(status != 1) // file reading failed
            {return 0;}
        
        return 1;
    }

    /* ----- ECAT Communication ----- */
    void EL5102::read_output_from_ECAT_buffer()
    {
        // Double mutex
        mtx_lock(m_output_mutex);
        pthread_mutex_lock(ECAT->m_ECAT_output_mutex);

        /* Store in temporary buffer */
        double channel1_position = channel1->m_gear_ratio*(( (double) ECAT->m_ECAT_outputs->channel1.counter_value)
            /channel1->m_pulses_per_revolution - channel1->m_revolution_offset);
        double channel2_position = channel2->m_gear_ratio*(( (double) ECAT->m_ECAT_outputs->channel2.counter_value)
            /channel2->m_pulses_per_revolution - channel2->m_revolution_offset);

        /* Get EtherCat time difference */
        double time_difference = ((double) (ec_DCtime - m_prev_DCtime))/1000000000;
        m_prev_DCtime = ec_DCtime;
        pthread_mutex_unlock(ECAT->m_ECAT_output_mutex);

        // Get position difference
        double position1_difference = channel1_position - channel1->m_outputs->position;        
        double position2_difference = channel2_position - channel2->m_outputs->position;        
        
        // Update outputs 
        channel1->m_outputs->position = channel1_position;
        channel2->m_outputs->position = channel2_position;

        /* Check if new sample was taken */
        if(time_difference > 0)
        {
            channel1->m_outputs->velocity_unfiltered = position1_difference/time_difference;
            channel2->m_outputs->velocity_unfiltered = position2_difference/time_difference;
        }
        
        channel1->filter_input_velocity();
        channel2->filter_input_velocity();

        mtx_unlock(m_output_mutex);
    }
    void EL5102::IO_channel_t::filter_input_velocity()
    {
        /* Apply filter */
        double velocity_filtered = m_outputs->velocity_unfiltered*low_pass_filter.b[0];
        for(int i = 0; i < 4; i++)  
        {
            velocity_filtered += low_pass_filter.v_meas[i]*low_pass_filter.b[i+1] 
                - low_pass_filter.v_filt[i]*low_pass_filter.a[i];
        }
        /* Shift buffer */
        for(int i = 0; i < 3; i++)  
        {
            low_pass_filter.v_meas[3-i] = low_pass_filter.v_meas[2-i];
            low_pass_filter.v_filt[3-i] = low_pass_filter.v_filt[2-i];
        }
        low_pass_filter.v_meas[0] = m_outputs->velocity_unfiltered;
        low_pass_filter.v_filt[0] = velocity_filtered;

        m_outputs->velocity = velocity_filtered;
    }

    void EL5102::write_input_to_ECAT_buffer()
    {
        // Double mutex
        mtx_lock(m_input_mutex);
        pthread_mutex_lock(ECAT->m_ECAT_input_mutex);

        ECAT->m_ECAT_inputs->channel1.set_counter = channel1->m_inputs->counter;
        ECAT->m_ECAT_inputs->channel2.set_counter = channel2->m_inputs->counter;
        ECAT->m_ECAT_inputs->channel1.set_counter_value = channel1->m_inputs->counter_value;
        ECAT->m_ECAT_inputs->channel2.set_counter_value = channel2->m_inputs->counter_value;

        pthread_mutex_unlock(ECAT->m_ECAT_input_mutex);
        mtx_unlock(m_input_mutex);
    }

    /* <-----------------> */
    /* <ECAT_EL5102 Class> */
    /* <-----------------> */
    ECAT_EL5102::ECAT_EL5102()
    {
        // set_state(-1);
        m_follower_name = "EL5102";
    }
    ECAT_EL5102::~ECAT_EL5102()
    {
        delete m_ECAT_inputs;
        delete m_ECAT_outputs;
    }

    int ECAT_EL5102::configure_cyclic_PDO_mapping(const char* input_file)
    {
        /* Set TxPDO and RxPDO datastructure */
        PDO::PDO_struct_t<2, 0x0a > inputmap;
        inputmap.length     = 0x02;
        inputmap.index[0]   = 0x1600;
        inputmap.index[1]   = 0x1606;

        PDO::PDO_struct_t<2, 0x16 > outputmap;
        outputmap.length     = 0x02;
        /* Channel 1 */
        outputmap.index[0]   = 0x1A00;
        // outputmap.index[1]   = 0x1A06;
        // outputmap.index[2]   = 0x1A08;
        // outputmap.index[3]   = 0x1A0A;
        // outputmap.index[4]   = 0x1A0B;
        /* Channel 2 */
        outputmap.index[1]   = 0x1A0D;
        // outputmap.index[6]   = 0x1A13;
        // outputmap.index[7]   = 0x1A15;
        // outputmap.index[8]   = 0x1A17;
        // outputmap.index[9]   = 0x1A18;

        if(PDO::PDO_set_and_check(m_follower_no, PDO_SYNC_MANAGER_2_RXPDO, 0x00, TRUE, sizeof(inputmap), inputmap, EC_TIMEOUTSAFE) != 1)
            {return 0;}
        if(PDO::PDO_set_and_check(m_follower_no, PDO_SYNC_MANAGER_3_TXPDO, 0x00, TRUE, sizeof(outputmap), outputmap, EC_TIMEOUTSAFE) != 1)
            {return 0;}

        /* Configuration Parameters */
        uint32 supply_voltage;
        uint32 filter_setting;
        uint32 counter_mode;   
        int pulses_per_revolution;

        /* Read from input file */
        int i = 0;
        param_array_t param_array[4];
        param_array[i] = (param_array_t) {"supply_voltage", &supply_voltage, PARAM_TYPE_INT,OPTIONAL_PARAMETER}; i++;
        param_array[i] = (param_array_t) {"filter_setting", &filter_setting, PARAM_TYPE_INT,OPTIONAL_PARAMETER}; i++;
        param_array[i] = (param_array_t) {"counter_mode", &counter_mode, PARAM_TYPE_INT}; i++;
        param_array[i] = (param_array_t) {"channel1/pulses_per_revolution", &pulses_per_revolution, PARAM_TYPE_INT}; i++;
        

        int status;
        read_from_input_file(input_file, param_array,i,  &status);
        if(status != 1) // file reading failed
            {return 0;}

        if(PDO::PDO_set_and_check(m_follower_no, 0xF008,0, FALSE,4,0x72657375,EC_TIMEOUTSAFE) != 1)
            {return 0;}
        usleep(100000);

        /* Make maximum pulses a multiple op pulses per revolution */
        // PDO::PDO_subindex_struct_t<1> encoder_settings_1_channel_1;
        // encoder_settings_1_channel_1.PDO = 0x8001;
        // encoder_settings_1_channel_1.subPDO[0] = {0x1A,(0xFFFFFFFF/pulses_per_revolution)*pulses_per_revolution,4};
        // if(PDO::PDO_array_set_and_check(m_follower_no, encoder_settings_1_channel_1, EC_TIMEOUTSAFE) != 1)
        //     {return 0;}

        /* Start in the middle of the pulses */
        // if(PDO::PDO_set_and_check(m_follower_no, 0x8012,1, FALSE,4,0x01,EC_TIMEOUTSAFE) != 1)
        //     {return 0;}
        

//         /* Use same values for both channels */
// This shit doesn't work yet :(
        // PDO::PDO_subindex_struct_t<1> encoder_settings_1_channel_1;
        // encoder_settings_1_channel_1.PDO = 0x8001;
        // encoder_settings_1_channel_1.subindex[0] = {0x17,supply_voltage,4};
        // encoder_settings_1_channel_1.subindex[1-1] = {0x19,filter_setting,4};
        // encoder_settings_1_channel_1.subindex[1-1] = {0x19,0x00001388,4};
        // encoder_settings_1_channel_1.subindex[0] = {0x1D,counter_mode,4};
        
//         PDO::PDO_subindex_struct_t<3> encoder_settings_1_channel_2;
//         encoder_settings_1_channel_2.PDO = 0x8011;
//         encoder_settings_1_channel_2.subindex[0] = {0x17,supply_voltage,4};
//         encoder_settings_1_channel_2.subindex[1] = {0x19,filter_setting,4};
//         encoder_settings_1_channel_2.subindex[2] = {0x1D,counter_mode,4};
// // 0x80n0:08 "Disable filter" = FALSE.
//         ec_SDOwrite(m_follower_no, 0x8001, 0x1D, FALSE, 2, &counter_mode, EC_TIMEOUTSAFE);
//         cout << "kl " << counter_mode << endl;
// if(PDO::PDO_set_and_check(m_follower_no, 0x8001,0x1D, FALSE,4,counter_mode,EC_TIMEOUTSAFE) != 1)
//             {return 0;}

        // if(PDO::PDO_array_set_and_check(m_follower_no, encoder_settings_1_channel_1, EC_TIMEOUTSAFE) != 1)
        //     {return 0;}
//         if(PDO::PDO_array_set_and_check(m_follower_no, encoder_settings_1_channel_2, EC_TIMEOUTSAFE) != 1)
//             {return 0;}

        return 1;
    }

    /* ----- RW ----- */
    void ECAT_EL5102::set_follower_name(const char* follower_name) {m_follower_name = follower_name;}
    const char* ECAT_EL5102::get_follower_name() const {return m_follower_name;}
    void ECAT_EL5102::set_follower_no(int follower_no) {m_follower_no = follower_no;}    
    const int ECAT_EL5102::get_follower_no() const {return m_follower_no;}
    int* ECAT_EL5102::get_follower_no_address() {return &m_follower_no;}
    
    void ECAT_EL5102::set_dc_time()
        {ec_dcsync0(m_follower_no, TRUE, 2000000, 0);}

    /* ----- SOEM Linking ----- */
    void ECAT_EL5102::assign_IO_with_SOEM()
    {
        /* !EtherCAT defines inputs and outputs the otherway around! */
        m_ECAT_inputs = (EL5102_ECAT_inputs_t*) ec_slave[m_follower_no].outputs;
        m_ECAT_outputs = (EL5102_ECAT_outputs_t*) ec_slave[m_follower_no].inputs; 

        /* Reroute mutexes from SOEM */
        m_ECAT_input_mutex = &((&ecx_context)->port->tx_mutex); 
        m_ECAT_output_mutex = &((&ecx_context)->port->rx_mutex); 
    }
    /* <\ECAT_EL5102 Class> */

} // namespace ecat
