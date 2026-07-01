// C++ _practice02
#include <iostream>
#include <string.h>
enum class MotorMode
{
    Idle,
    Position,
    Torque,
    Error
};

struct MotorState
{
    int id;
    std::string name;
    double measured_position;
    double measured_velocity;
    double target_position;
    bool online;
    MotorMode mode;
};

bool target_inspector(const MotorState &motor, double position_min, double position_max)
{
    if (motor.online == true)
    {
        std::cout << "motor is online, proceeding to next ... " << '\n';
        if (motor.mode == MotorMode::Error)
        {
            std::cout << "motor is online, but encountered error, shut down .. " << '\n';
            return false;
        }
        else if (motor.target_position > position_max || motor.target_position < position_min)
        {
            std::cout << "motor exceeds position limit, shut down ... " << '\n';
            return false;
        }
        else
        {
            std::cout << "motor is safe, online, no error, within limit." << '\n';
            return true;
        }
    }
    else
    {
        std::cout << "motor is offline, aborting.. " << '\n';
        return false;
    }
}

int main()
{
    double position_max = 45.02;
    double position_min = 23.32;
    const MotorState motor1 = {
        1, "motor1", 32.32, 32.232, 50.23, true, MotorMode::Position};

    target_inspector(motor1, position_min, position_max);

    return 0;
}