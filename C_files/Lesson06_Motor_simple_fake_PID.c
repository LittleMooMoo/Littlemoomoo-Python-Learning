// Lesson06 Motor_simple PID(Fake)
#include <stdio.h>
#include <string.h>
#include <math.h>
#include "Lesson06_Math_helper.h"

double motor_current_position;
double motor_target_position;
double motor_move_time;
float kp = 0.2;
float kd = 0.005;
char motor_name[30] = "front_right_shank";
int motor_id = 1;
const double pi = 3.14159265389793;
char move_confirmation[30];

int main()
{
    printf("please enter motor current position: ");
    scanf("%lf", &motor_current_position);
    printf("please enter motor target position: ");
    scanf("%lf", &motor_target_position);
    printf("please enter motor move time: ");
    scanf("%lf", &motor_move_time);
    printf("motor current position is: %.3lf \n", motor_current_position);
    printf("motor target position is :%.3lf \n", motor_target_position);
    printf("motor move time is : %.3lf \n", motor_move_time);
    printf("calculating position command .....\n");

    double error = calculated_error(motor_target_position, motor_current_position);
    double error_step = calculated_error_step(0, error);
    double damping = calculated_damping(error_step, motor_move_time, kd);
    double spring_force = calculated_drive_force(error, kp);

    double position_command = -damping + spring_force + error * motor_move_time;
    double motor_trajectory = 0.5 * (1 - cos(2 * pi * motor_move_time));

    printf("calculation finished, please type y to confirm move or q to quit: \n ");
    scanf("%29s", move_confirmation);

    if (strcmp(move_confirmation, "y") == 0)
    {
        printf("motor with id %d is moving towards target : %lf ...\n", motor_id, position_command);
        printf("motor with name '%s' is moving with trajectory point: %lf ...\n", motor_name, motor_trajectory);
    }
    else if (strcmp(move_confirmation, "q") == 0)
    {
        printf("user quit");
    }
    else
    {
        printf("invalid input");
    }

    return 0;
}
