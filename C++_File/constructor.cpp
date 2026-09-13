#include "constructor.h"
#include <stdio.h>
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

int main()
{
    motor dingmotor(1, 2);
    motor *pdingmotor = &dingmotor;
    float pos = pdingmotor->getState();
    pos = pos + 1000;
    dingmotor.getState() = pos;
    printf("pos: %f", dingmotor.getState());
}