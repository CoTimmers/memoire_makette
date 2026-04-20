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

            /* Operator / attach position */
            double attach_position_x;       // [mm]
            double attach_position_y;       // [mm]
            double attach_position_z;       // [mm]

            /* Steady state detection (slide 5) */
            double steady_state_threshold_px;   // epsilon_d en pixels
            int    steady_state_n_frames;        // nombre de frames consécutives (≥60 @ 30fps)

            /* Velocity profile + position feedback (slides 3 & 4) */
            double L_cable;         // longueur câble [m], ≈ 2.2
            double v_max;           // vitesse max [mm/s]
            double kp_position;     // gain proportionnel erreur → v_ref
            double dt;              // période d'échantillonnage [s]
            double epsilon_x;       // tolérance position [mm]
            double epsilon_v;       // tolérance vitesse [mm/s]

            /* Target correction (slide 7) */
            double correction_step_mm;  // pas de correction vers le mur [mm], default 20mm

        } params_t;

        /* FSM states */
        typedef enum gantry_state_e
        {
            GANTRY_HOMING               = 0,
            GANTRY_WAITING_OPERATOR     = 1,
            GANTRY_WAITING_STEADY       = 2,
            GANTRY_MOVING               = 3,
            GANTRY_CHECK_ALIGNMENT      = 4,
            GANTRY_ADJUST_POSITION      = 5,
            GANTRY_FINAL_POSITIONING    = 6,
            GANTRY_CORNERING            = 7,
            GANTRY_DONE                 = 8
        } gantry_state_t;

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
            double target_x;    // [mm]
            double target_y;    // [mm]

            /* Velocity profile state (slides 3 & 4) */
            double v_ref_prev;  // v_ref du cycle précédent pour limitation a_max

            /* Camera data — written by camera thread, read by FSM */
            double marker_pixel_x;
            double marker_pixel_y;
            bool   marker_detected;

            /* Steady state detection (slide 5) */
            double marker_prev_pixel_x;
            double marker_prev_pixel_y;
            int    steady_state_frame_count;

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

            /* Homing */
            uint HOMING_COMPLETE        : 1;

            /* External signals */
            uint OPERATOR_COMPLETE      : 1;    // mis à 1 par l'opérateur (touche Enter)

            /* Adjustment loop flag —
             * mis à 1 par check_alignment quand FLAG_ALIGNED == 0
             * pour que waiting_steady_state retourne à CHECK_ALIGNMENT
             * et non à MOVING après stabilisation */
            uint COMING_FROM_ADJUSTMENT : 1;

            /* Alignment flag (slide 6)
             * 0 = aucun côté aligné
             * 1 = côté long aligné  → GANTRY_FINAL_POSITIONING
             * 2 = côté court aligné → GANTRY_CORNERING          */
            int FLAG_ALIGNED;

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

        params_t             *params;
        continuous_state_t   *continuous_state;
        discrete_state_t     *discrete_state;
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
        void do_nothing(void *in);
    };

} // namespace activity_5c

#endif // ECAT_ACTIVITY_HPP