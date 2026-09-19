class motor
{
public:
    enum mode
    {
        POS,
        VEL,
        MIT
    };
    struct state
    {
        float pos;
        float vel;
    };
    motor(int id, int dir);
    float getState();
    motor::mode getmode();

    motor::state get_motor_state();

private:
    state motor_state_;
    mode mode_;
    int can_id;
    int direction;
};