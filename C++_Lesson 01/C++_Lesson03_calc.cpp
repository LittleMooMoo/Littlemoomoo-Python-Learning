#include <iostream>

int main()
{
    double motor_kp = 0.2;
    double motor_kd = 0.005;
    double target = 100;
    double measured = 30;
    double error = target - measured;
    double cycle_time = 0.002;

    double command = -(motor_kp * error + motor_kd * (0.35 * error / cycle_time) + cycle_time * error);

    std::cout << "error = " << error << "\n";
    std::cout << "command = " << command << "\n";
}