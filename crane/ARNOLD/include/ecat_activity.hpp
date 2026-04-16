/* ----------------------------------------------------------------------------
 * Project Title,
 * ROB @ KU Leuven, Leuven, Belgium
 * See LICENSE for the license information
 * -------------------------------------------------------------------------- */

/**
 * @file ecat_activity.hpp
 * @date July 6, 2023
 **/

#ifndef ECAT_ACTIVITY_HPP
#define ECAT_ACTIVITY_HPP

#include <stdio.h>
#include <time.h>
#include <pthread.h>

/* AACAL */
#include <five_c/activity/activity.h>

/* Ethercat modules */
#include "beckhoff_modules/EL5102.hpp"
#include "beckhoff_modules/EL7047.hpp"
#include "beckhoff_modules/EL1002.hpp"

#include "epos_SOEM_driver/epos_driver.hpp"

/* Read file */
#include <read_file/read_file.h>

namespace activity_5c
{
    class ECATActivity
    {
    private:
    public:
        activity_t *activity;

        ECATActivity();
        ~ECATActivity();

        // ! parameters
        typedef struct params_s
        {
            char configuration_file[256];

            /* Configuration Files */
            char path_to_configuration_file_motor_x[128];
            char path_to_configuration_file_motor_y[128];
            char path_to_configuration_file_motor_z[128];
            char path_to_configuration_file_encoder[128];
            char FSM_schedule[128];
            char ifname[128];

            int motor_x_number_in_chain;
            int motor_y_number_in_chain;

            int RT_thread_cycletime;

            int RT_sync_jitter;

            /* Gantry application parameters */
            double attach_position_x;   // Position to move to for operator to attach bac (mm)
            double attach_position_y;   // Position to move to for operator to attach bac (mm)
            double attach_position_z;   // Position to move to for operator to attach bac (mm)
            double correction_step_mm;  // Step size for position correction (mm), default 10mm = 1cm
            double steady_state_threshold_px; // Max pixel movement to consider bac steady
            int    steady_state_n_frames;     // Number of consecutive frames below threshold
        } params_t;

        /* FSM states */
        typedef enum gantry_state_e
        {
            GANTRY_HOMING           = 0,
            GANTRY_MOVE_TO_ATTACH   = 1,
            GANTRY_WAITING_OPERATOR = 2,
            GANTRY_WAITING_STEADY   = 3,
            GANTRY_MOVING           = 4,
            GANTRY_CHECK_ALIGNMENT  = 5,
            GANTRY_ADJUST_POSITION  = 6,
            GANTRY_DONE             = 7
        } gantry_state_t;

        /* Alignment side */
        typedef enum alignment_side_e
        {
            SIDE_UNKNOWN = 0,
            SIDE_LONG    = 1,
            SIDE_SHORT   = 2
        } alignment_side_t;

        /* (Computational) continuous state */
        typedef struct continuous_state_s
        {
            ecat::EL7047    *motor_x;
            ecat::EL7047    *motor_y;
            ecat::EposDrive *motor_z;

            ecat::EL5102    *encoder;
            ecat::EL1002    *limit_switch;

            struct timespec *RT_ts;
            void (*execute_task_function)(void *in);
            void (*execute_FSM)(void *in);

            /* Target position for move tasks */
            double target_x;    // mm
            double target_y;    // mm

            /* Camera data — written by camera thread, read by FSM */
            double marker_pixel_x;          // pixel x of green square centre
            double marker_pixel_y;          // pixel y of green square centre
            bool   marker_detected;         // true if marker was found this frame

            /* Steady state data */
            double marker_prev_pixel_x;     // previous frame position
            double marker_prev_pixel_y;
            int    steady_state_frame_count; // consecutive frames below threshold

            /* Alignment result — written by camera thread during SCANNING/CHECK */
            bool             wall_aligned;       // true if bac side is aligned with wall
            alignment_side_t alignment_side;     // which side is aligned (LONG or SHORT)

        } continuous_state_t;

        /* (Computational) discrete state */
        typedef struct discrete_state_s
        {
            /* EtherCAT internal flags */
            uint ETHERCAT_CONFIGURED : 1;
            uint ETHERCAT_NEW_DATA   : 1;
            uint EXECUTE_RT_THREAD   : 1;
            uint ETHERCAT_RUNNING    : 1;

            int RT_SYNCED;

            /* External signals — written by threads outside the FSM */
            uint OPERATOR_COMPLETE  : 1;   // set by operator thread when Enter pressed
            uint SCANNING_COMPLETE  : 1;   // set by camera thread when scan is done

            /* Main FSM state */
            gantry_state_t gantry_state;

        } discrete_state_t;

        /* Coordination state */
        typedef struct coordination_state_s
        {
            bool *execution_request;
            bool *deinitialisation_request;
            bool *configuration_request;
        } coordination_state_t;

        params_t           *params;
        continuous_state_t *continuous_state;
        discrete_state_t   *discrete_state;
        coordination_state_t *coordination_state;

        /* ----- Functions ----- */
        void RT_thread();
        void start_RT_thread();

        /* Five-C */
        void activity_config();

        void creation_compute();
        void creation_coordinate();
        void creation_configure();
        void creation();

        int  resource_configuration_compute();
        void resource_configuration_coordinate();
        void resource_configuration_configure();
        void resource_configuration();

        void capability_configuration_communicate();
        void capability_configuration_coordinate();
        void capability_configuration_configure();
        void capability_configuration_compute();
        void capability_configuration();

        void running_compute();
        void running_coordinate();
        void running_configure();
        void running();

        /* Scheduler */
        void register_schedules();

        /* Activity LCSM */
        void create_lcsm();
        void resource_configure_lcsm();
        void destroy_lcsm();

        /* Configuration */
        void configure_from_file(const char *file_path, int *status);

        /* Execution functions */
        static void do_nothing(void *in);
    };

} // namespace activity_5c

#endif // ECAT_ACTIVITY_HPP
