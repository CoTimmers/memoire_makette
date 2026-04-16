#ifndef TASK_FSM_HPP
#define TASK_FSM_HPP

namespace ARNOLD{
    /* Original homing FSM — kept for reference */
    void homing_FSM(void *in);

    /* Main gantry FSM — replaces homing_FSM as execute_FSM */
    void gantry_FSM(void *in);
}

#endif /* TASK_FSM_HPP */
