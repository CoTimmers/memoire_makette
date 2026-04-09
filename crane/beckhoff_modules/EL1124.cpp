/* 
    Beckhoff EL1124, 

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
#include "beckhoff_modules/EL1124.hpp"

#include "SOEM_helper_functions/etherCAT_communication.h"
#include "SOEM_helper_functions/soem_mapping.hpp"
#include "read_file/read_file.h"

using namespace std;

namespace ecat
{
    EL1124::EL1124()
    {        
        m_inputs = new struct inputs_s;
        
        m_outputs = new struct outputs_s;
        
        m_input_mutex = new pthread_mutex_t;
        m_output_mutex = new pthread_mutex_t;

        ECAT = new ecat::ECAT_EL1124;
    }
    EL1124::~EL1124()
    {
        delete m_inputs;

        delete m_outputs;
        
        delete m_input_mutex;
        delete m_output_mutex;
        delete ECAT;
    }

    int EL1124::initialise()
    {
        /* Link correct follower id */
        if(write_follower_number(ECAT->get_follower_name(),  
            ECAT->get_follower_no_address()) == 0)
            {return 0;}

        /* Set TxPDO and RxPDO mapping */
        if(ECAT->configure_cyclic_PDO_mapping() != 1)
            {return 0;}

        return 1;
    }

    int EL1124::configure()
    {
        if(go_to_SAFEOP(ECAT->get_follower_no()) == 0)
            {return 0;}

        /* connect struct pointers to slave I/O pointers, SOEM defines I/O from follower perspective */
        ECAT->assign_IO_with_SOEM();

        if(go_to_OP(ECAT->get_follower_no(), 1000000000) == 0)
            {return 0;}

        return 1;
    }



    /* ----- RW ----- */
    // void EL1124::set_input_file(const char* t_input_file)
    //     {strcpy(m_input_file,t_input_file);}

    /* ----- ECAT Communication ----- */
    void EL1124::read_output_from_ECAT_buffer()
    {
        // Double mutex
        pthread_mutex_lock(m_output_mutex);
        pthread_mutex_lock(ECAT->m_ECAT_output_mutex);

        pthread_mutex_unlock(ECAT->m_ECAT_output_mutex);
        pthread_mutex_unlock(m_output_mutex);
    }

    void EL1124::write_input_to_ECAT_buffer()
    {
         // Double mutex
        pthread_mutex_lock(m_input_mutex);
        pthread_mutex_lock(ECAT->m_ECAT_input_mutex);

        pthread_mutex_unlock(ECAT->m_ECAT_input_mutex);
        pthread_mutex_unlock(m_input_mutex);
    }

    /* <-----------------> */
    /* <ECAT_EL1124 Class> */
    /* <-----------------> */



    ECAT_EL1124::ECAT_EL1124()
    {
        // set_state(-1);
        m_follower_name = "EL1124";
    }
    ECAT_EL1124::~ECAT_EL1124()
    {
        delete m_ECAT_inputs;
        delete m_ECAT_outputs;
    }

    /* ----- RW ----- */
    void ECAT_EL1124::set_follower_name(const char* follower_name) {m_follower_name = follower_name;}
    const char* ECAT_EL1124::get_follower_name() const {return m_follower_name;}
    void ECAT_EL1124::set_follower_no(int follower_no) {m_follower_no = follower_no;}    
    const int ECAT_EL1124::get_follower_no() const {return m_follower_no;}
    int* ECAT_EL1124::get_follower_no_address() {return &m_follower_no;}
    
    void ECAT_EL1124::set_dc_time()
        {ec_dcsync0(m_follower_no, TRUE, 2000000, 0);}

    int ECAT_EL1124::configure_cyclic_PDO_mapping()
    {
        /* Set TxPDO and RxPDO datastructure */
        PDO::PDO_struct_t<4, 0x16 > outputmap;
        outputmap.length     = 0x04;
        /* Channel 1 */
        outputmap.index[0]   = 0x1A00;
        outputmap.index[1]   = 0x1A01;
        outputmap.index[2]   = 0x1A02;
        outputmap.index[3]   = 0x1A03;

        // if(PDO::PDO_set_and_check(m_follower_no, PDO_SYNC_MANAGER_3_TXPDO, 0x00, TRUE, sizeof(outputmap), outputmap, EC_TIMEOUTSAFE) != 1)
        //     {return 0;}

        return 1;
    }
    
    /* ----- SOEM Linking ----- */
    void ECAT_EL1124::assign_IO_with_SOEM()
    {
        /* !EtherCAT defines inputs and outputs the otherway around! */
        m_ECAT_inputs = (EL1124_ECAT_inputs_t*) ec_slave[m_follower_no].outputs;
        m_ECAT_outputs = (EL1124_ECAT_outputs_t*) ec_slave[m_follower_no].inputs; 

        /* Reroute mutexes from SOEM */
        m_ECAT_input_mutex = &((&ecx_context)->port->tx_mutex); 
        m_ECAT_output_mutex = &((&ecx_context)->port->rx_mutex); 
    }
    /* <\ECAT_EL1124 Class> */

} // namespace ecat
