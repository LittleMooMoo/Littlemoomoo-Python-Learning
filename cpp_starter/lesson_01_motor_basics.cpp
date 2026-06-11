// Lesson 1: C++ motor basics for a robotics beginner.
//
// This file is a C++ version of ideas you have already seen in Python:
// variables, functions, if statements, loops, classes, and objects.
//
// A C++ file usually ends with .cpp.
// Unlike Python, C++ must be compiled before it runs.

// iostream gives us std::cout for terminal output.
#include <iostream>

// string gives us std::string for text.
#include <string>

// vector gives us std::vector, which is similar to a Python list.
#include <vector>

// This function limits a motor command to a safe range.
// double means a decimal number, similar to Python's float.
double clamp_motor_command(double command, double minimum, double maximum) {
    // If the command is too low, return the minimum safe value.
    if (command < minimum) {
        return minimum;
    }

    // If the command is too high, return the maximum safe value.
    if (command > maximum) {
        return maximum;
    }

    // If the command is already safe, return it unchanged.
    return command;
}

// A class is a blueprint for creating objects.
// This Motor class stores one actuator's CAN ID, name, and command.
class Motor {
public:
    // int stores a whole number.
    int can_id;

    // std::string stores text.
    std::string name;

    // double stores a decimal number.
    double command;

    // This is a constructor.
    // It runs when a Motor object is created.
    Motor(int input_can_id, std::string input_name) {
        can_id = input_can_id;
        name = input_name;
        command = 0.0;
    }

    // This method safely sets the motor command.
    void set_command(double raw_command) {
        command = clamp_motor_command(raw_command, -1.0, 1.0);
    }

    // This method prints the motor's information.
    void describe() {
        std::cout << name << " | CAN ID: " << can_id
                  << " | command: " << command << "\n";
    }
};

// main is where a simple C++ program starts.
int main() {
    std::cout << "C++ Lesson 1: Motor Basics\n\n";

    // Create three motor objects.
    Motor hip_front_left(3, "hip_front_left");
    Motor thigh_front_left(2, "thigh_front_left");
    Motor shank_front_left(1, "shank_front_left");

    // Set motor commands.
    hip_front_left.set_command(0.5);
    thigh_front_left.set_command(1.8);
    shank_front_left.set_command(-1.4);

    // Store the motors in a vector.
    std::vector<Motor> motors = {
        hip_front_left,
        thigh_front_left,
        shank_front_left,
    };

    // A for loop repeats once for every motor in the vector.
    for (Motor motor : motors) {
        motor.describe();
    }

    std::cout << "\nSearching for CAN ID 2...\n";

    // Search for one motor by CAN ID.
    int searched_can_id = 2;
    bool found = false;

    for (Motor motor : motors) {
        if (motor.can_id == searched_can_id) {
            std::cout << "Found motor: " << motor.name << "\n";
            found = true;
            break;
        }
    }

    if (!found) {
        std::cout << "No motor found.\n";
    }

    // return 0 means the program finished successfully.
    return 0;
}
