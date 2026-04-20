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

    // =========================================================
    // NOUVELLES FONCTIONS
    // =========================================================

    /* ----------------------------------------------------------
     * WAITING OPERATOR
     * Moteurs désactivés. On attend que l'opérateur signale
     * que le bac est accroché (OPERATOR_COMPLETE = 1).
     * ---------------------------------------------------------- */
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

    /* ----------------------------------------------------------
     * WAITING STEADY STATE  (slide 5 — COM motion tracking)
     *
     * Deux destinations possibles selon COMING_FROM_ADJUSTMENT :
     *   0 → GANTRY_MOVING        (premier mouvement vers le mur)
     *   1 → GANTRY_CHECK_ALIGNMENT (retour après correction)
     * ---------------------------------------------------------- */
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
                // Retour après correction → on vérifie l'alignement à nouveau
                ecat_activity->discrete_state->COMING_FROM_ADJUSTMENT = 0;
                ecat_activity->discrete_state->gantry_state =
                    activity_5c::ECATActivity::GANTRY_CHECK_ALIGNMENT;
            }
            else
            {
                // Premier mouvement → on avance vers le mur
                ecat_activity->discrete_state->gantry_state =
                    activity_5c::ECATActivity::GANTRY_MOVING;
            }
        }
    }

    /* ----------------------------------------------------------
     * MOVING  (slide 3 — velocity profile generator
     *        + slide 4 — position feedback)
     * ---------------------------------------------------------- */
    void moving(void *in)
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
        motor_z->set_input(0.0);

        // --- Paramètres dynamiques (slide 3) ---
        const double g         = 9.81;
        const double L         = ecat_activity->params->L_cable;
        const double T         = 2.0 * M_PI * sqrt(L / g);
        const double theta_max = 0.07;                         // 4° en rad
        const double alpha     = 0.7;
        const double a_max     = alpha * g * theta_max;        // ≈ 0.48 m/s²
        const double v_max     = ecat_activity->params->v_max;
        const double t_j       = T / 10.0;
        const double j_max     = a_max / t_j;

        // --- Erreur de position (slide 4) ---
        double target_x = ecat_activity->continuous_state->target_x;
        double pos_x    = encoder->channel1->get_position();
        double error_x  = target_x - pos_x;

        // --- v_ref proportionnel à l'erreur, saturé à v_max (slide 4) ---
        double v_ref = error_x * ecat_activity->params->kp_position;
        if(v_ref >  v_max) v_ref =  v_max;
        if(v_ref < -v_max) v_ref = -v_max;

        // --- Limitation de l'accélération : |dv/dt| <= a_max (slide 4) ---
        double v_prev = ecat_activity->continuous_state->v_ref_prev;
        double dt     = ecat_activity->params->dt;
        double dv     = v_ref - v_prev;
        double dv_max = a_max * dt * 1000.0;  // m/s² → mm/s par cycle
        if(dv >  dv_max) dv =  dv_max;
        if(dv < -dv_max) dv = -dv_max;
        v_ref = v_prev + dv;
        ecat_activity->continuous_state->v_ref_prev = v_ref;

        motor_x->set_enable(1);
        motor_x->set_input(v_ref);
        motor_y->set_enable(0);
        motor_y->set_input(0);

        // --- Flag "position atteinte" (slide 4) ---
        double vel_x = encoder->channel1->get_velocity();
        if(fabs(error_x) <= ecat_activity->params->epsilon_x &&
           fabs(vel_x)   <= ecat_activity->params->epsilon_v)
        {
            motor_x->set_enable(0);
            motor_x->set_input(0);
            ecat_activity->continuous_state->steady_state_frame_count = 0;
            ecat_activity->continuous_state->v_ref_prev = 0.0;
            ecat_activity->discrete_state->gantry_state =
                activity_5c::ECATActivity::GANTRY_CHECK_ALIGNMENT;
        }
    }

    /* ----------------------------------------------------------
     * CHECK ALIGNMENT  (slide 6 — alignment verification)
     *
     * FLAG_ALIGNED mis à jour par le thread caméra :
     *   0 = aucun côté aligné → correction + retour à MOVING via ADJUST_POSITION
     *   1 = côté long aligné  → GANTRY_FINAL_POSITIONING
     *   2 = côté court aligné → GANTRY_CORNERING
     * ---------------------------------------------------------- */
    void check_alignment(void *in)
    {
        activity_5c::ECATActivity* ecat_activity = (activity_5c::ECATActivity*) in;

        ecat_activity->continuous_state->motor_x->set_enable(0);
        ecat_activity->continuous_state->motor_x->set_input(0);
        ecat_activity->continuous_state->motor_y->set_enable(0);
        ecat_activity->continuous_state->motor_y->set_input(0);

        if(ecat_activity->discrete_state->FLAG_ALIGNED == 1)
        {
            ecat_activity->discrete_state->gantry_state =
                activity_5c::ECATActivity::GANTRY_FINAL_POSITIONING;
        }
        else if(ecat_activity->discrete_state->FLAG_ALIGNED == 2)
        {
            ecat_activity->discrete_state->gantry_state =
                activity_5c::ECATActivity::GANTRY_CORNERING;
        }
        else
        {
            // Aucun côté aligné : décale la cible de correction_step_mm vers le mur
            // COMING_FROM_ADJUSTMENT = 1 pour que waiting_steady_state
            // retourne à CHECK_ALIGNMENT et non à MOVING
            ecat_activity->continuous_state->target_x +=
                ecat_activity->params->correction_step_mm;
            ecat_activity->continuous_state->v_ref_prev          = 0.0;
            ecat_activity->discrete_state->COMING_FROM_ADJUSTMENT = 1;
            ecat_activity->discrete_state->gantry_state =
                activity_5c::ECATActivity::GANTRY_ADJUST_POSITION;
        }
    }

    /* ----------------------------------------------------------
     * ADJUST POSITION  (slide 7 — target correction)
     *
     * La nouvelle target_x a été mise à jour dans check_alignment.
     * On délègue entièrement le déplacement à moving() qui gère
     * le profil de vitesse et transite vers GANTRY_CHECK_ALIGNMENT
     * une fois la position atteinte.
     * ---------------------------------------------------------- */
    void adjust_position(void *in)
    {
        moving(in);
    }

    /* ----------------------------------------------------------
     * FINAL POSITIONING — TODO
     * Côté long aligné. À implémenter.
     * ---------------------------------------------------------- */
    void final_positioning(void *in)
    {
        activity_5c::ECATActivity* ecat_activity = (activity_5c::ECATActivity*) in;
        // TODO
    }

    /* ----------------------------------------------------------
     * CORNERING — TODO
     * Côté court aligné. À implémenter.
     * ---------------------------------------------------------- */
    void cornering(void *in)
    {
        activity_5c::ECATActivity* ecat_activity = (activity_5c::ECATActivity*) in;
        // TODO
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