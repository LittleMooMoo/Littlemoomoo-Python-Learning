#include "constructor.h"
#include <stdio.h>
#include <Eigen/Dense>

motor::motor(int id, int dir)
    : can_id{id},
      direction{dir}
{
}

float motor::getState()
{
    return motor_state_.pos;
}
motor::mode motor::getmode()
{
    return mode_;
}

motor::state motor::get_motor_state()
{
    return motor_state_.pos;
}

float get_x_vel(float t)
{
    float x = std::sin(t);
    return x;
}

Eigen::Vector3d calculate_vel(double t)
{
    Eigen::Vector3d vel_vector;
    vel_vector << pow(t, 2),
        t,
        1;
    return vel_vector;
}
int main()
{
    motor dingmotor(1, 2);
    motor *pdingmotor = &dingmotor;
    float pos = pdingmotor->getState();
    pos = pos + 1000;
    dingmotor.getState() = pos;
    printf("pos: %f", dingmotor.getState());
}