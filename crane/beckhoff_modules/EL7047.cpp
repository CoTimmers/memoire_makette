/* 
    Beckhoff EL7047, STEP Drive.

    Author: Boris Deroo 
    KU Leuven, ROB group 
*/

#include <stdio.h>
#include <string.h>
#include <inttypes.h>
#include <unistd.h>
#include <stdlib.h>
#include <iostream>
#include <threads.h>

#include "ethercat.h"
#include "beckhoff_modules/EL7047.hpp"

#include "SOEM_helper_functions/etherCAT_communication.h"
#include "SOEM_helper_functions/soem_mapping.hpp"
#include "read_file/read_file.h"

using namespace std;

namespace ecat
{
    EL7047::EL7047()
    {        
        m_inputs = new struct inputs_s;
        m_outputs = new struct outputs_s;
        m_motor_constants = new struct motor_constants_s;
        m_input_mutex = new mtx_t;
        m_output_mutex = new mtx_t;

        mtx_init(m_input_mutex, mtx_plain );
        mtx_init(m_output_mutex, mtx_plain );

        ECAT = new ecat::ECAT_EL7047;
    }
    EL7047::~EL7047()
    {
        delete m_inputs;
        delete m_outputs;
        delete m_motor_constants;
        delete m_input_mutex;
        delete m_output_mutex;
        delete ECAT;
    }

    int EL7047::initialise()
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

    int EL7047::initialise(int follower_number_in_chain)
    {
        /* Link correct follower id */
        if(write_follower_number_specified(ECAT->get_follower_name(),  
            ECAT->get_follower_no_address(),follower_number_in_chain) == 0)
            {return 0;}
        
        /* Set TxPDO and RxPDO mapping */
        if(ECAT->configure_cyclic_PDO_mapping(m_input_file) != 1)
            {return 0;}

        return 1;
    }

    int EL7047::configure()
    {
        /* Config motor */
        if(config_motor() != 1)
            {return 0;}

        if(go_to_SAFEOP(ECAT->get_follower_no()) == 0)
            {return 0;}

        /* connect struct pointers to slave I/O pointers, SOEM defines I/O from follower perspective */
        ECAT->assign_IO_with_SOEM();

        if(go_to_OP(ECAT->get_follower_no(), 1000000000) == 0)
            {return 0;}

        return 1;
    }

    // /* ----- I/O RW ----- */
    void EL7047::set_input_file(const char* t_input_file)
        {strcpy(m_input_file,t_input_file);}
    int EL7047::set_control_mode(int t_mode)
    {
        if( (t_mode != EL7047_VELOCITY_CONTROL) && (t_mode != EL7047_POSITION_CONTROL) )
        {
            cout << "ERROR: Invalid control mode: " << t_mode << endl;
            return 0;
        }
        else
        {
            mtx_lock(m_input_mutex);
            m_inputs->control_mode = t_mode;
            mtx_unlock(m_input_mutex);

            return 1;
        }
    }
    int EL7047::get_control_mode() const
    {
        mtx_lock(m_input_mutex);
        int t_mode = m_inputs->control_mode;
        mtx_unlock(m_input_mutex);
        return t_mode;
    }
    void EL7047::set_input(double t_input)
    {
        mtx_lock(m_input_mutex);
        m_inputs->input = t_input;
        mtx_unlock(m_input_mutex);
    }
    double EL7047::get_input() const
    {
        mtx_lock(m_input_mutex);
        double t_input = m_inputs->input;
        mtx_unlock(m_input_mutex);
        return t_input;
    }
    void EL7047::set_enable(double t_input)
    {
        mtx_lock(m_input_mutex);
        m_inputs->enable = t_input;
        mtx_unlock(m_input_mutex);
    }
    int EL7047::get_enable() const
    {
        mtx_lock(m_input_mutex);
        int t_input = m_inputs->enable;
        mtx_unlock(m_input_mutex);
        return t_input;
    }
    double EL7047::get_position() const
    {
        mtx_lock(m_output_mutex);
        double t_position = m_outputs->internal_position;
        mtx_unlock(m_output_mutex);
        return t_position;
    }
    double EL7047::get_velocity() const
    {
        mtx_lock(m_output_mutex);
        double t_velocity = m_outputs->velocity;
        mtx_unlock(m_output_mutex);
        return t_velocity;
    }
    // float EL7047::get_input_voltage() const
    // {
    //     mtx_lock(m_output_mutex);
    //     double t_input_voltage = m_outputs->input_voltage;
    //     mtx_unlock(m_output_mutex);
    //     return t_input_voltage;
    // }

    /* ----- I/O Adresses ----- */
    mtx_t* EL7047::get_input_mutex() const
        {return m_input_mutex;}
    int* EL7047::get_control_mode_address() const
        {return &(m_inputs->control_mode);}

    mtx_t* EL7047::get_output_mutex() const
        {return m_output_mutex;}
    
    /* ----- ECAT Communication ----- */
    void EL7047::read_output_from_ECAT_buffer()
    {
        // Double mutex
        mtx_lock(m_output_mutex);
        pthread_mutex_lock(ECAT->m_ECAT_output_mutex);
        
        m_outputs->velocity = ECAT->m_ECAT_outputs->position_status.actual_velocity;
        m_outputs->internal_position = ECAT->m_ECAT_outputs->internal_position;

        pthread_mutex_unlock(ECAT->m_ECAT_output_mutex);
        mtx_unlock(m_output_mutex);
    }

    void EL7047::write_input_to_ECAT_buffer()
    {
        double input;
        
        // Double mutex
        mtx_lock(m_input_mutex);
        pthread_mutex_lock(ECAT->m_ECAT_input_mutex);

        /* Overwrite enable if killswitch is active */
        if(*flags.KILLSWITCH_ENGAGED == 1)
            {flags.STO_TRIGGERED = 1;}
        if(flags.STO_TRIGGERED == 1)
        {
            m_inputs->enable = 0;
            (ECAT->m_ECAT_inputs->control.enable) = 0;
        }
        else
            {(ECAT->m_ECAT_inputs->control.enable) = m_inputs->enable;}

        /* Check if control mode changed, since then drive needs to reconfigure */
        if(m_inputs->control_mode == EL7047_VELOCITY_CONTROL)
        {
            input = m_inputs->input/m_motor_constants->velocity_factor; 
            (ECAT->m_ECAT_inputs->input) = ((int16_t) input);
        }
        /* TO DO */
        // POSITION CONTROL 

        // else
            // {printf("ERROR: No valid control method (%i) in write_to_drive_function!\n", m_inputs->control_mode);}

        pthread_mutex_unlock(ECAT->m_ECAT_input_mutex);
        mtx_unlock(m_input_mutex);
    }

    int EL7047::config_motor()
    {   
        int speed_range;

        /* Read from input file */
        int i = 0;  // Keeps tracks of amount of elements inside param_array
        int number_of_parameters = 3;
        
        param_array_t param_array[number_of_parameters];

        param_array[i] = (param_array_t) {"motor_fullsteps", &(m_motor_constants->motor_fullsteps), PARAM_TYPE_DOUBLE}; i++;
        param_array[i] = (param_array_t) {"speed_range", &speed_range, PARAM_TYPE_INT}; i++;
        param_array[i] = (param_array_t) {"gear_ratio", &(m_motor_constants->gear_ratio), PARAM_TYPE_DOUBLE}; i++;
        
        int status;
        read_from_input_file(m_input_file, param_array,i,  &status);
        if(status != 1) // file reading failed
            {return 0;}
            
        if(speed_range == 0)
            {m_motor_constants->reference_velocity = 1000;} 
        else if(speed_range == 1)
            {m_motor_constants->reference_velocity = 2000;} 
        else if(speed_range == 2)
            {m_motor_constants->reference_velocity = 4000;} 
        else if(speed_range == 3)
            {m_motor_constants->reference_velocity = 8000;} 
        else if(speed_range == 4)
            {m_motor_constants->reference_velocity = 16000;} 
        else if(speed_range == 5)
            {m_motor_constants->reference_velocity = 32000;} 
        else
        {
            cout << "ERROR: Invalid speed range specified in '" << m_input_file << "'!" << endl;
            return 0;
        }

        /* Conversion factors to interpret etherCAT data */
        m_motor_constants->velocity_factor =  (m_motor_constants->gear_ratio*m_motor_constants->reference_velocity)/(m_motor_constants->motor_fullsteps)/((double) m_max_speed_integer);
        // m_motor_constants->position_factor = 1/( m_motor_constants->gear_ratio*m_motor_constants->encoder_resolution );
        return 1;
    }

    /* <--------------------> */
    /* <ECAT_EL7047 Class> */
    /* <--------------------> */
    ECAT_EL7047::ECAT_EL7047()
    {
        // set_state(-1);
        m_follower_name = "EL7047";
    }
    ECAT_EL7047::~ECAT_EL7047()
    {
        delete m_ECAT_inputs;
        delete m_ECAT_outputs;
    }

    int ECAT_EL7047::configure_cyclic_PDO_mapping(const char* input_file)
    {
        /* Set Datastructure for Sync Managers */
        /* RxPDO assignment for velocity mode */
        PDO::PDO_struct_t<3, 0x0a > inputmap;
        inputmap.length     = 0x03;
        inputmap.index[0]   = 0x1601;
        inputmap.index[1]   = 0x1602;
        inputmap.index[2]   = 0x1604;

        /* TxPDO assignment */
        PDO::PDO_struct_t<5, 0x14 > outputmap;
        outputmap.length     = 0x05;
        // outputmap.index[0]   = 0x1a03;
        // outputmap.index[1]   = 0x1a07;
        // outputmap.index[2]   = 0x1a08;
        // outputmap.index[3]   = 0x1a09;
        // outputmap.index[4]   = 0x1a04;  // info data 1 & 2, has to be set through 0x8012

        outputmap.index[0]   = 0x1a03;
        outputmap.index[1]   = 0x1a04;
        outputmap.index[2]   = 0x1a07;
        outputmap.index[3]   = 0x1a08;
        outputmap.index[4]   = 0x1a09;  // info data 1 & 2, has to be set through 0x8012

        if(PDO::PDO_set_and_check(m_follower_no, PDO_SYNC_MANAGER_2_RXPDO, 0x00, TRUE, sizeof(inputmap), inputmap, EC_TIMEOUTSAFE) != 1)
            {return 0;}
        if(PDO::PDO_set_and_check(m_follower_no, PDO_SYNC_MANAGER_3_TXPDO, 0x00, TRUE, sizeof(outputmap), outputmap, EC_TIMEOUTSAFE) != 1)
            {return 0;}

        /* Configuration parameters */
        /* Parameters relevant to velocity and position control are send */
        uint max_current;
        uint reduced_current = 2500; 
        uint nominal_voltage;
        uint motor_resistance;
        uint motor_fullsteps;
        uint motor_inductance;
        uint drive_on_delay_time = 100;
        uint drive_off_delay_time = 150;

        uint Kp_current_controller = 150;
        uint Ki_current_controller = 10;
        
        uint operation_mode = 0;
        uint speed_range = 1;
        uint feedback_type;
        uint invert_motor_direction;

        /* Read from input file */
        int i = 0;
        param_array_t param_array[14];

        param_array[i] = (param_array_t) {"max_current", &max_current, PARAM_TYPE_INT}; i++;
        param_array[i] = (param_array_t) {"reduced_current", &reduced_current, PARAM_TYPE_INT, OPTIONAL_PARAMETER}; i++;
        param_array[i] = (param_array_t) {"nominal_voltage", &nominal_voltage, PARAM_TYPE_INT}; i++;
        param_array[i] = (param_array_t) {"motor_resistance", &motor_resistance, PARAM_TYPE_INT}; i++;
        param_array[i] = (param_array_t) {"motor_fullsteps", &motor_fullsteps, PARAM_TYPE_INT}; i++;
        param_array[i] = (param_array_t) {"motor_inductance", &motor_inductance, PARAM_TYPE_INT}; i++;
        param_array[i] = (param_array_t) {"drive_on_delay_time", &drive_on_delay_time, PARAM_TYPE_INT,OPTIONAL_PARAMETER}; i++;
        param_array[i] = (param_array_t) {"drive_off_delay_time", &drive_off_delay_time, PARAM_TYPE_INT,OPTIONAL_PARAMETER}; i++;
        
        param_array[i] = (param_array_t) {"Kp_current_controller", &Kp_current_controller, PARAM_TYPE_INT,OPTIONAL_PARAMETER}; i++;
        param_array[i] = (param_array_t) {"Ki_current_controller", &Ki_current_controller, PARAM_TYPE_INT,OPTIONAL_PARAMETER}; i++;
        
        param_array[i] = (param_array_t) {"operation_mode", &operation_mode, PARAM_TYPE_INT,OPTIONAL_PARAMETER}; i++;
        param_array[i] = (param_array_t) {"speed_range", &speed_range, PARAM_TYPE_INT,OPTIONAL_PARAMETER}; i++;
        param_array[i] = (param_array_t) {"feedback_type", &feedback_type, PARAM_TYPE_INT}; i++;
        param_array[i] = (param_array_t) {"invert_motor_direction", &invert_motor_direction, PARAM_TYPE_INT}; i++;
        
        int status;
        read_from_input_file(input_file, param_array,i,  &status);
        if(status != 1) // file reading failed
            {return 0;}

        PDO::PDO_subindex_struct_t<8> motor_settings_1;
        motor_settings_1.PDO = 0x8010;
        motor_settings_1.subPDO[0] = {0x01,max_current,0x10};
        motor_settings_1.subPDO[1] = {0x02,reduced_current,0x10};
        motor_settings_1.subPDO[2] = {0x03,nominal_voltage,0x10};
        motor_settings_1.subPDO[3] = {0x04,motor_resistance,0x10};
        motor_settings_1.subPDO[4] = {0x06,motor_fullsteps,0x10};
        motor_settings_1.subPDO[5] = {0x0A,motor_inductance,0x10};
        motor_settings_1.subPDO[6] = {0x10,drive_on_delay_time,0x10};
        motor_settings_1.subPDO[7] = {0x11,drive_off_delay_time,0x10};

        PDO::PDO_subindex_struct_t<2> controller_settings_1;
        controller_settings_1.PDO = 0x8011;
        controller_settings_1.subPDO[0] = {0x01,Kp_current_controller,0x10};
        controller_settings_1.subPDO[1] = {0x02,Ki_current_controller,0x10};

uint info_data_1 = 13;
uint info_data_2 = 104;

        PDO::PDO_subindex_struct_t<6> features_1;
        features_1.PDO = 0x8012;
        features_1.subPDO[0] = {0x01,operation_mode,0x08};
        features_1.subPDO[1] = {0x05,speed_range,0x08};
        features_1.subPDO[2] = {0x08,feedback_type,0x08};
        features_1.subPDO[3] = {0x09,invert_motor_direction,0x08}; 
        features_1.subPDO[4] = {0x11,info_data_1,0x08}; 
        features_1.subPDO[5] = {0x19,info_data_2,0x08}; 

        if(PDO::PDO_array_set_and_check(m_follower_no, motor_settings_1, EC_TIMEOUTSAFE) != 1)
            {return 0;}
        if(PDO::PDO_array_set_and_check(m_follower_no, controller_settings_1, EC_TIMEOUTSAFE) != 1)
            {return 0;}
        if(PDO::PDO_array_set_and_check(m_follower_no, features_1, EC_TIMEOUTSAFE) != 1)
            {return 0;}

        return 1;
    }

    /* ----- RW ----- */
    void ECAT_EL7047::set_follower_name(const char* follower_name) {m_follower_name = follower_name;}
    const char* ECAT_EL7047::get_follower_name() const {return m_follower_name;}
    void ECAT_EL7047::set_follower_no(int follower_no) {m_follower_no = follower_no;}    
    const int ECAT_EL7047::get_follower_no() const {return m_follower_no;}
    int* ECAT_EL7047::get_follower_no_address() {return &m_follower_no;}
    

    void ECAT_EL7047::set_dc_time()
        {ec_dcsync0(m_follower_no, TRUE, 2000000, 0);}

    /* ----- SOEM Linking ----- */
    void ECAT_EL7047::assign_IO_with_SOEM()
    {
        /* !EtherCAT defines inputs and outputs the otherway around! */
        m_ECAT_inputs = (EL7047_ECAT_inputs_t*) ec_slave[m_follower_no].outputs;
        m_ECAT_outputs = (EL7047_ECAT_outputs_t*) ec_slave[m_follower_no].inputs; 

        /* Reroute mutexes from SOEM */
        m_ECAT_input_mutex = &((&ecx_context)->port->tx_mutex);  
        m_ECAT_output_mutex = &((&ecx_context)->port->rx_mutex); 
    }
    /* <\ECAT_EL7047 Class> */

} // namespace ecat
