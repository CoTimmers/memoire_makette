/**
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

#include "ARNOLD_gantry/task_application.hpp"
#include "ARNOLD_gantry/ecat_activity.hpp"

#include "beckhoff_modules/EL7047.hpp"
#include "beckhoff_modules/EL5102.hpp"
#include "beckhoff_modules/EL1002.hpp"

#include "time_functions/time_functions.h"
#include "SOEM_helper_functions/etherCAT_communication.h"

#include <iostream>

using namespace std;

namespace ARNOLD{

    /* Semantic meaning of digital IO */
    int get_limit_switch_x(ecat::EL1002* read_module)
        {return !read_module->read_input1();}
    int get_limit_switch_y(ecat::EL1002* read_module)
        {return !read_module->read_input2();}

    /* Task functions */
    void do_nothing(void *in)
    {}

    void homing(void *in)
    {
        activity_5c::ECATActivity* ecat_activity = (activity_5c::ECATActivity*) in;

        ecat::EL5102* encoder      = ecat_activity->continuous_state->encoder;
        ecat::EL1002* limit_switch = ecat_activity->continuous_state->limit_switch;
        ecat::EL7047* motor_x      = ecat_activity->continuous_state->motor_x;
        ecat::EL7047* motor_y      = ecat_activity->continuous_state->motor_y;
        ecat::EposDrive* motor_z   = ecat_activity->continuous_state->motor_z;

        motor_x->set_control_mode(EL7047_VELOCITY_CONTROL);
        motor_y->set_control_mode(EL7047_VELOCITY_CONTROL);
        motor_z->set_control_mode(EPOS_VELOCITY_CONTROL);
        motor_z->set_brake(0);
        motor_z->set_input(.0);

        int calibrated = 0;
        double velocity_fast = -8;
        double velocity_slow = -1;
        double velocity_x, velocity_y;
        int homing_complete_x = 0;
        int homing_complete_y = 0;

        if(get_limit_switch_x(limit_switch) == 1)
        {
            encoder->channel1->set_counter(1);
            encoder->channel1->set_counter_value(0);
            motor_x->set_enable(0);
            motor_x->set_input(0);
            homing_complete_x = 1;
        }
        else
        {
            if(encoder->channel1->get_position() > 1.5)
                {velocity_x = velocity_fast;}
            else
                {velocity_x = velocity_slow;}
            motor_x->set_enable(1);
            motor_x->set_input(velocity_x);
        }

        if(get_limit_switch_y(limit_switch) == 1)
        {
            encoder->channel2->set_counter(1);
            encoder->channel2->set_counter_value(0);
            motor_y->set_enable(0);
            motor_y->set_input(0);
            homing_complete_y = 1;
        }
        else
        {
            if(encoder->channel2->get_position() > 1.5)
                {velocity_y = velocity_fast;}
            else
                {velocity_y = velocity_slow;}
            motor_y->set_enable(1);
            motor_y->set_input(velocity_y);
        }

        if((homing_complete_x == 1) && (homing_complete_y == 1))
            {ecat_activity->discrete_state->HOMING_COMPLETE = 1;}
    }

    void move(void *in)
    {
        activity_5c::ECATActivity* ecat_activity = (activity_5c::ECATActivity*) in;

        ecat::EL5102* encoder    = ecat_activity->continuous_state->encoder;
        ecat::EL7047* motor_x    = ecat_activity->continuous_state->motor_x;
        ecat::EL7047* motor_y    = ecat_activity->continuous_state->motor_y;
        ecat::EposDrive* motor_z = ecat_activity->continuous_state->motor_z;

        motor_x->set_control_mode(EL7047_VELOCITY_CONTROL);
        motor_y->set_control_mode(EL7047_VELOCITY_CONTROL);
        motor_z->set_control_mode(EPOS_VELOCITY_CONTROL);
        motor_z->set_brake(0);
        motor_z->set_input(-.0);

        double velocity_fast = 20;
        double velocity_slow = 1;
        double velocity_x, velocity_y;
        int move_complete_x = 0;
        int move_complete_y = 0;

        if(encoder->channel1->get_position() < 100)
        {
            if(encoder->channel1->get_position() < 98.5)
                {velocity_x = velocity_fast;}
            else
                {velocity_x = velocity_slow;}
            motor_x->set_enable(1);
            motor_x->set_input(velocity_x);
        }
        else
        {
            move_complete_x = 1;
            motor_x->set_enable(0);
            motor_x->set_input(0);
        }

        if(encoder->channel2->get_position() < 60)
        {
            if(encoder->channel2->get_position() < 58.5)
                {velocity_y = velocity_fast;}
            else
                {velocity_y = velocity_slow;}
            motor_y->set_enable(1);
            motor_y->set_input(velocity_y);
        }
        else
        {
            move_complete_y = 1;
            motor_y->set_enable(0);
            motor_y->set_input(0);
        }

        if((move_complete_x == 1) && (move_complete_y == 1))
            {ecat_activity->discrete_state->HOMING_COMPLETE = 0;}
    }


    void waiting_operator(void *in)
    {
        activity_5c::ECATActivity* ecat_activity = (activity_5c::ECATActivity*) in;

        ecat_activity->continuous_state->motor_x->set_enable(0);
        ecat_activity->continuous_state->motor_x->set_input(0);
        ecat_activity->continuous_state->motor_y->set_enable(0);
        ecat_activity->continuous_state->motor_y->set_input(0);
        ecat_activity->continuous_state->motor_z->set_input(0);

        if(ecat_activity->discrete_state->OPERATOR_COMPLETE == 1)
        {
            ecat_activity->discrete_state->OPERATOR_COMPLETE      = 0;
            ecat_activity->discrete_state->COMING_FROM_ADJUSTMENT = 0;
            ecat_activity->continuous_state->steady_state_frame_count = 0;
            ecat_activity->discrete_state->gantry_state =
                activity_5c::ECATActivity::GANTRY_WAITING_STEADY;
        }
    }

    void waiting_steady_state(void *in)
    {
        activity_5c::ECATActivity* ecat_activity = (activity_5c::ECATActivity*) in;

        ecat_activity->continuous_state->motor_x->set_enable(0);
        ecat_activity->continuous_state->motor_x->set_input(0);
        ecat_activity->continuous_state->motor_y->set_enable(0);
        ecat_activity->continuous_state->motor_y->set_input(0);

        if(!ecat_activity->continuous_state->marker_detected)
            {return;}

        double dx = ecat_activity->continuous_state->marker_pixel_x
                  - ecat_activity->continuous_state->marker_prev_pixel_x;
        double dy = ecat_activity->continuous_state->marker_pixel_y
                  - ecat_activity->continuous_state->marker_prev_pixel_y;
        double d = sqrt(dx*dx + dy*dy);

        ecat_activity->continuous_state->marker_prev_pixel_x =
            ecat_activity->continuous_state->marker_pixel_x;
        ecat_activity->continuous_state->marker_prev_pixel_y =
            ecat_activity->continuous_state->marker_pixel_y;

        if(d <= ecat_activity->params->steady_state_threshold_px)
            {ecat_activity->continuous_state->steady_state_frame_count++;}
        else
            {ecat_activity->continuous_state->steady_state_frame_count = 0;}

        if(ecat_activity->continuous_state->steady_state_frame_count
           >= ecat_activity->params->steady_state_n_frames)
        {
            ecat_activity->continuous_state->steady_state_frame_count = 0;

            if(ecat_activity->discrete_state->COMING_FROM_ADJUSTMENT == 1)
            {
                ecat_activity->discrete_state->COMING_FROM_ADJUSTMENT = 0;
                ecat_activity->discrete_state->gantry_state =
                    activity_5c::ECATActivity::GANTRY_CHECK_ALIGNMENT;
            }
            else
            {
                ecat_activity->discrete_state->gantry_state =
                    activity_5c::ECATActivity::GANTRY_MOVING;
            }
        }
    }

    static double eval_velocity_profile(double t, double sign,
                                        double a, double v_peak,
                                        double t1, double t2, double tf,
                                        bool trapezoidal)
    {
        if(t < 0)
            {return 0.0;}

        if(trapezoidal)
        {
            if(t <= t1)
                {return sign * a * t;}
            else if(t <= t2)
                {return sign * v_peak;}
            else if(t <= tf)
                {return sign * v_peak - sign * a * (t - t2);}
            else
                {return 0.0;}
        }
        else /* triangle */
        {
            double half = tf / 2.0;
            if(t <= half)
                {return sign * a * t;}
            else if(t <= tf)
                {return sign * v_peak - sign * a * (t - half);}
            else
                {return 0.0;}
        }
    }

    void moving(void *in)
    {
        activity_5c::ECATActivity* ecat_activity = (activity_5c::ECATActivity*) in;

        ecat::EL5102*    encoder = ecat_activity->continuous_state->encoder;
        ecat::EL7047*    motor_x = ecat_activity->continuous_state->motor_x;
        ecat::EL7047*    motor_y = ecat_activity->continuous_state->motor_y;
        ecat::EposDrive* motor_z = ecat_activity->continuous_state->motor_z;

        motor_x->set_control_mode(EL7047_VELOCITY_CONTROL);
        motor_y->set_control_mode(EL7047_VELOCITY_CONTROL);
        motor_z->set_control_mode(EPOS_VELOCITY_CONTROL);
        motor_z->set_brake(0);
        motor_z->set_input(0.0);

        if(!ecat_activity->continuous_state->trajectory_planned)
        {
            const double g    = 9810.0;                                  // mm/s²
            const double L    = ecat_activity->params->L_cable * 1000.0; // m → mm
            const double zeta = ecat_activity->params->zeta;
            const double wn   = sqrt(g / L);
            const double wd   = wn * sqrt(1.0 - zeta * zeta);
            const double half_T = M_PI / wd;                             // T/2 [s]

            const double v_max = ecat_activity->params->v_max;           // [mm/s]
            const double a_max = v_max / half_T;                         // [mm/s²]

            double x0 = encoder->channel1->get_position();
            double y0 = encoder->channel2->get_position();
            double Dx = ecat_activity->continuous_state->target_x - x0;
            double Dy = ecat_activity->continuous_state->target_y - y0;

            ecat_activity->continuous_state->x0 = x0;
            ecat_activity->continuous_state->y0 = y0;

            /* Axe X */
            double Dx_abs = fabs(Dx);
            if(Dx_abs >= v_max * half_T)    // cas trapézoïdal
            {
                ecat_activity->continuous_state->trapezoidal_x = true;
                ecat_activity->continuous_state->a_x      = a_max;
                ecat_activity->continuous_state->v_peak_x = v_max;
                ecat_activity->continuous_state->t1_x     = half_T;
                ecat_activity->continuous_state->t2_x     = Dx_abs / v_max;
                ecat_activity->continuous_state->tf_x     = ecat_activity->continuous_state->t2_x + half_T;
            }
            else                            // cas triangulaire
            {
                double a = Dx_abs / (half_T * half_T);
                ecat_activity->continuous_state->trapezoidal_x = false;
                ecat_activity->continuous_state->a_x      = a;
                ecat_activity->continuous_state->v_peak_x = a * half_T;
                ecat_activity->continuous_state->t1_x     = half_T;
                ecat_activity->continuous_state->t2_x     = half_T;
                ecat_activity->continuous_state->tf_x     = 2.0 * half_T;
            }

            /* Axe Y */
            double Dy_abs = fabs(Dy);
            if(Dy_abs >= v_max * half_T)    // cas trapézoïdal
            {
                ecat_activity->continuous_state->trapezoidal_y = true;
                ecat_activity->continuous_state->a_y      = a_max;
                ecat_activity->continuous_state->v_peak_y = v_max;
                ecat_activity->continuous_state->t1_y     = half_T;
                ecat_activity->continuous_state->t2_y     = Dy_abs / v_max;
                ecat_activity->continuous_state->tf_y     = ecat_activity->continuous_state->t2_y + half_T;
            }
            else                            // cas triangulaire
            {
                double a = Dy_abs / (half_T * half_T);
                ecat_activity->continuous_state->trapezoidal_y = false;
                ecat_activity->continuous_state->a_y      = a;
                ecat_activity->continuous_state->v_peak_y = a * half_T;
                ecat_activity->continuous_state->t1_y     = half_T;
                ecat_activity->continuous_state->t2_y     = half_T;
                ecat_activity->continuous_state->tf_y     = 2.0 * half_T;
            }

            struct timespec ts_now;
            clock_gettime(CLOCK_MONOTONIC, &ts_now);
            ecat_activity->continuous_state->t_start =
                ts_now.tv_sec + ts_now.tv_nsec * 1e-9;

            ecat_activity->continuous_state->trajectory_planned = true;
        }

        struct timespec ts_now;
        clock_gettime(CLOCK_MONOTONIC, &ts_now);
        double t_now = ts_now.tv_sec + ts_now.tv_nsec * 1e-9;
        double t = t_now - ecat_activity->continuous_state->t_start;

        double Dx = ecat_activity->continuous_state->target_x
                  - ecat_activity->continuous_state->x0;
        double Dy = ecat_activity->continuous_state->target_y
                  - ecat_activity->continuous_state->y0;

        double sign_x = (Dx >= 0) ? 1.0 : -1.0;
        double sign_y = (Dy >= 0) ? 1.0 : -1.0;

        double v_ref_x = eval_velocity_profile(t, sign_x,
            ecat_activity->continuous_state->a_x,
            ecat_activity->continuous_state->v_peak_x,
            ecat_activity->continuous_state->t1_x,
            ecat_activity->continuous_state->t2_x,
            ecat_activity->continuous_state->tf_x,
            ecat_activity->continuous_state->trapezoidal_x);

        double v_ref_y = eval_velocity_profile(t, sign_y,
            ecat_activity->continuous_state->a_y,
            ecat_activity->continuous_state->v_peak_y,
            ecat_activity->continuous_state->t1_y,
            ecat_activity->continuous_state->t2_y,
            ecat_activity->continuous_state->tf_y,
            ecat_activity->continuous_state->trapezoidal_y);

        motor_x->set_enable(1);
        motor_x->set_input(v_ref_x);
        motor_y->set_enable(1);
        motor_y->set_input(v_ref_y);

        double tf_max = fmax(ecat_activity->continuous_state->tf_x,
                             ecat_activity->continuous_state->tf_y);
        if(t >= tf_max)
        {
            motor_x->set_enable(0);
            motor_x->set_input(0);
            motor_y->set_enable(0);
            motor_y->set_input(0);
            ecat_activity->continuous_state->trajectory_planned = false;
            ecat_activity->continuous_state->steady_state_frame_count = 0;
            ecat_activity->discrete_state->gantry_state =
                activity_5c::ECATActivity::GANTRY_WAITING_STEADY;
        }
    }

    void check_alignment(void *in)
    {
        activity_5c::ECATActivity* ecat_activity = (activity_5c::ECATActivity*) in;

        ecat_activity->continuous_state->motor_x->set_enable(0);
        ecat_activity->continuous_state->motor_x->set_input(0);
        ecat_activity->continuous_state->motor_y->set_enable(0);
        ecat_activity->continuous_state->motor_y->set_input(0);

        if(!ecat_activity->continuous_state->marker_detected)
            return;

        double y_ref   = ecat_activity->params->frame_height_px / 2.0;
        double y_cam   = ecat_activity->continuous_state->marker_pixel_y;
        double e_crate = y_ref - y_cam;

        if(fabs(e_crate) <= ecat_activity->params->epsilon_crate_tol_px)
        {
            if(ecat_activity->discrete_state->FLAG_ALIGNED == 1)
                ecat_activity->discrete_state->gantry_state =
                    activity_5c::ECATActivity::GANTRY_DONE;
            // else: wait for camera thread to update FLAG_ALIGNED
        }
        else
        {
            double delta_x = ecat_activity->params->kp_camera * e_crate;
            ecat_activity->continuous_state->target_x           += delta_x;
            ecat_activity->continuous_state->trajectory_planned  = false;
            ecat_activity->discrete_state->COMING_FROM_ADJUSTMENT = 1;
            ecat_activity->discrete_state->gantry_state =
                activity_5c::ECATActivity::GANTRY_ADJUST_POSITION;
        }
    }

    void adjust_position(void *in)
    {
        moving(in);
    }

    void done(void *in)
    {
        activity_5c::ECATActivity* ecat_activity = (activity_5c::ECATActivity*) in;

        ecat_activity->continuous_state->motor_x->set_enable(0);
        ecat_activity->continuous_state->motor_x->set_input(0);
        ecat_activity->continuous_state->motor_y->set_enable(0);
        ecat_activity->continuous_state->motor_y->set_input(0);
        ecat_activity->continuous_state->motor_z->set_input(0.0);
    }
}
