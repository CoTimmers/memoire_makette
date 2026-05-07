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

            /* Steady state detection */
            double steady_state_threshold_px;   // epsilon_d  [pixels]
            int    steady_state_n_frames;        // nombre de frames consécutives (≥60 @ 30fps)

            /* Trajectory generation — input shaping */
            double L_cable;         // [m]
            double zeta;            // [-]
            double v_max;           //  [mm/s]

            /* Target correction */
            double correction_step_mm;  // pas de correction [mm]

            double frame_height_px;       // hauteur du frame caméra [px] (= b)
            double epsilon_crate_tol_px;  // tolérance alignement caméra [px]
            double kp_camera;             // gain discret: Delta_x = kp_camera * e_crate

            double x_wall_px;             // position x du mur 1 [px] (calibré manuellement)
            double epsilon_wall_px;       // seuil contact coin-mur [px]
            double D_threshold_px;        // seuil long/court côté [px] (~35cm en pixels)

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
            GANTRY_DONE                 = 6
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

            /* Trajectory planning */
            bool   trajectory_planned;
            double t_start;         // temps absolu début du mouvement [s]
            double x0;              // position X au début du mouvement [mm]
            double y0;              // position Y au début du mouvement [mm]

            double a_x;             // accélération axe X [mm/s²]
            double v_peak_x;        // vitesse pic axe X [mm/s]
            double t1_x;            // fin phase accél X [s]
            double t2_x;            // début phase décel X [s]
            double tf_x;            // fin du mouvement X [s]
            bool   trapezoidal_x;   // true = trapèze, false = triangle

            double a_y;             // accélération axe Y [mm/s²]
            double v_peak_y;        // vitesse pic axe Y [mm/s]
            double t1_y;            // fin phase accél Y [s]
            double t2_y;            // début phase décel Y [s]
            double tf_y;            // fin du mouvement Y [s]
            bool   trapezoidal_y;   // true = trapèze, false = triangle

            /* Camera data — written by camera thread */
            double marker_pixel_x;
            double marker_pixel_y;
            bool   marker_detected;

            /* Steady state detection */
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
            uint OPERATOR_COMPLETE      : 1;    // by pressing 'enter

            /* mis à 1 par check_alignment après correction caméra
             * pour que waiting_steady_state retourne à CHECK_ALIGNMENT */
            uint COMING_FROM_ADJUSTMENT : 1;

            /* Alignment flag
             * 0 = non aligné
             * 1 = aligné → GANTRY_DONE          */
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
