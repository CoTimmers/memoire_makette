#ifndef TASK_APPLICATION_HPP
#define TASK_APPLICATION_HPP

namespace ARNOLD{

    int get_limit_switch_x(void *in);
    int get_limit_switch_y(void *in);

    void do_nothing(void *in);
    void homing(void *in);
    void waiting_operator(void *in);
    void waiting_steady_state(void *in);
    void moving(void *in);
    void check_alignment(void *in);
    void adjust_position(void *in);
    void done(void *in);
}

#endif /* TASK_APPLICATION_HPP */
