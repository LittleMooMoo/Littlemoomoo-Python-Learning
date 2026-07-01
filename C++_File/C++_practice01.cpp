// C++ _practice01
#include <iostream>
#include <cstring>

enum class Motor_Mode
{
    Idle,
    Position,
    Torque,
    Error,
};

std::string moode_to_string(Motor_Mode mode)
{
    if (mode == Motor_Mode::Idle)
    {
        return "Idle";
    }
    if (mode == Motor_Mode::Position)
    {
        return "Position";
    }
    if (mode == Motor_Mode::Torque)
    {
        return "Torque";
    }
    if (mode == Motor_Mode::Error)
    {
        return "Error";
    }
    return "Unknown";
}

Motor_Mode string_to_mode(std::string choice)
{
    if (choice == "Idle")
    {
        return Motor_Mode::Idle;
    }

    if (choice == "Position")
    {
        return Motor_Mode::Position;
    }
    if (choice == "Torque")
    {
        return Motor_Mode::Torque;
    }
    if (choice == "Error")
    {
        return Motor_Mode::Error;
    }
    return Motor_Mode::Error;
}

int main()
{
    std::string typed_choice;
    std::cout << "Please enter your selected mode: ";
    std::cin >> typed_choice;

    Motor_Mode mode = string_to_mode(typed_choice);

    std::cout << "here is the converted motor mode : " << moode_to_string(mode) << '\n';

    return 0;
}
