#ifndef TASK_APPLICATION_HPP
#define TASK_APPLICATION_HPP

namespace ARNOLD{

    /* Limit switch helpers */
    int get_limit_switch_x(void *in);
    int get_limit_switch_y(void *in);

    /* Task functions */
    void do_nothing(void *in);

    /* Phase 1 — Homing (existing) */
    void homing(void *in);

    /* Phase 2 — Move to attach position */
    void move_to_attach(void *in);

    /* Phase 3 — Wait for operator to attach bac */
    void waiting_operator(void *in);

    /* Phase 4 — Scanning (gantry immobile, camera detects bac + walls) */
    void scanning(void *in);

    /* Phase 5 — Wait for bac to reach steady state */
    void waiting_steady_state(void *in);
    void moving(void *in);

    /* Phase 6 — Check alignment of bac with wall */
    void check_alignment(void *in);

    /* Phase 7 — Adjust gantry position by 1cm */
    void adjust_position(void *in);

    /* Phase 8 — Done */
    void done(void *in);
}

#endif /* TASK_APPLICATION_HPP */
