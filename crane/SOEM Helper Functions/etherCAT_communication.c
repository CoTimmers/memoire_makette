#include <stdio.h>
#include <string.h>
#include <inttypes.h>
#include <unistd.h>

// #include "time.h"
#include "ethercat.h"
#include "SOEM_helper_functions/etherCAT_communication.h"
#include "time_functions/time_functions.h"

// #include "ax_5203_idn_values.h"
char IOmap[4096];

int init_ethercat(const char *ifname)
{    
    printf("init_ethercat\n");
    /* initialise SOEM, bind socket to ifname */
    if (ec_init(ifname))
    {
        printf("ec_init on %s succeeded.\n",ifname);
        /* find and auto-config slaves */
        if ( ec_config_init(FALSE) > 0 )
        {
            printf("%d slaves found\n",ec_slavecount);
            for(int i = 1; i<=ec_slavecount; i++)
                {printf("Slave %d: %s\n",i, ec_slave[i].name);}
        }
        else
            {return 0;}
    }
    else
        {return 0;}

    return 1;
}

void ecat_configure()
{
    ec_configdc();
    ec_config_map(&IOmap);
}

int go_to_PREOP(int follower_no)
{
    struct timespec ts, t_end;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    t_end = ts;                                 // Sync ns of current time
    t_end.tv_sec = ts.tv_sec + 10;             // Drive has 10 sec to get to SAFEOP

    int state = -1;
    int IN_PREOP = 0;
    while((state != EC_STATE_PRE_OP)&&(t_end.tv_sec > ts.tv_sec))
    {
        ec_slave[follower_no].state = EC_STATE_PRE_OP;
        ec_writestate(follower_no);
        state = ec_statecheck(follower_no,EC_STATE_PRE_OP,  EC_TIMEOUTSTATE);
        if(state == EC_STATE_PRE_OP){IN_PREOP = 1;}
        clock_gettime(CLOCK_MONOTONIC, &ts);
    }
    if(IN_PREOP == 0)
        {printf("Time out, PREOP not reached for follower: '%s'!\n", ec_slave[follower_no].name);
        return 0;}
    else
        {printf("PREOP reached for follower %s\n", ec_slave[follower_no].name);
        return 1;}
}

int go_to_SAFEOP(int follower_no)
{
    struct timespec ts, t_end;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    t_end = ts;                                 // Sync ns of current time
    t_end.tv_sec = ts.tv_sec + 10;             // Drive has 10 sec to get to SAFEOP

    int state = -1;
    int IN_SAFEOP = 0;
    while((state != EC_STATE_SAFE_OP)&&(t_end.tv_sec > ts.tv_sec))
    {
        ec_slave[follower_no].state = EC_STATE_SAFE_OP;
        ec_writestate(follower_no);
        state = ec_statecheck(follower_no,EC_STATE_SAFE_OP,  EC_TIMEOUTSTATE);
        if(state == EC_STATE_SAFE_OP){IN_SAFEOP = 1;}
        clock_gettime(CLOCK_MONOTONIC, &ts);
    }
    if(IN_SAFEOP == 0)
        {printf("Time out, SAFEOP not reached for follower: '%s' in position %i!\n", ec_slave[follower_no].name, follower_no);
        return 0;}
    else
        {printf("SAFEOP reached for follower %s in position %i\n", ec_slave[follower_no].name, follower_no);
        return 1;}
}

int go_to_OP(int follower_no, int timeout)
{
    // Set constants 
    if(ec_slave[follower_no].state == EC_STATE_SAFE_OP){
        printf("Request operational state for follower: '%s'\n", ec_slave[follower_no].name);
        ec_slave[follower_no].state = EC_STATE_OPERATIONAL;

        /* send one valid process data to make outputs in slaves happy*/
        ec_send_processdata();
        ec_receive_processdata(EC_TIMEOUTRET);

        /* request OP state for all slaves */
        ec_writestate(follower_no);

        struct timespec ts, t_timer;
        clock_gettime(CLOCK_MONOTONIC, &t_timer);
        add_timespec(&t_timer, timeout);   
        do
        {
            ec_statecheck(follower_no, EC_STATE_OPERATIONAL, EC_TIMEOUTSTATE);
            clock_gettime(CLOCK_MONOTONIC, &ts);
        }
        while( (ec_slave[follower_no].state != EC_STATE_OPERATIONAL) && (compare_time(ts,t_timer) != 1));

        if(ec_slave[follower_no].state == EC_STATE_OPERATIONAL)
        {
            printf("Operational state reached for follower: '%s'\n", ec_slave[follower_no].name);
            return 1;
        }
        else
        {
            printf("Operational state not reached for follower: '%s'\n", ec_slave[follower_no].name);
            ec_readstate();
            printf("Slave %d State=0x%2.2x StatusCode=0x%4.4x : %s\n",
            follower_no, ec_slave[follower_no].state, ec_slave[follower_no].ALstatuscode, ec_ALstatuscode2string(ec_slave[follower_no].ALstatuscode));
            return 0;
        }
    }

    return 0;
}

void go_to_init_and_shutdown()
{
    printf("Request init state for all slaves\n");
    ec_slave[0].state = EC_STATE_INIT;
    ec_writestate(0);
    ec_close();
}

int write_follower_number(const char *follower_name, int *follower_no)
{
    for(int i = 1; i<=ec_slavecount; i++){
        if(strcmp(ec_slave[i].name,follower_name) == 0)
        {
            *follower_no = i;
            return 1;
        }
    }

    printf("Error: No ECAT follower named '%s'!\n", follower_name);
    return 0;
} 

int write_follower_number_specified(const char *follower_name, int *follower_no, int number_in_chain)
{
    int found_followers = 0;
    for(int i = 1; i<=ec_slavecount; i++){
        if(strcmp(ec_slave[i].name,follower_name) == 0)
        {
            found_followers++; 
            if(found_followers == number_in_chain)
            {
                *follower_no = i;
                return 1;
            }
        }
    }

    printf("Error: No %i ECAT followers named '%s', last found '%s' is in position %i!\n", 
        number_in_chain, follower_name, follower_name, found_followers);
    return 0;
} 

void write_idn(int slave, int drive_no, int idn, int idnsize, const void *value)
{
    int wkc = 0;
    void* temp = (void*) value; 
    while(wkc == 0)
    {
        wkc = ec_SoEwrite(slave, drive_no, EC_SOE_VALUE_B, idn, idnsize, temp, EC_TIMEOUTRXM);
        if(wkc == 0)
        {
            printf("Error writing idn %04x, re-attempting to write \n", idn);         
        }
    }
}


