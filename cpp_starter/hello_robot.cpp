// C++ starter file for a beginner robotics programmer.
//
// C++ comments can start with //.
// The compiler ignores comments.
//
// Compile later with something like:
//
//     New-Item -ItemType Directory -Force build
//     g++ cpp_starter\hello_robot.cpp -std=c++17 -Wall -Wextra -o build\hello_robot.exe
//
// Then run:
//
//     .\build\hello_robot.exe

// #include brings in C++ library features.
#include <iostream>

// string lets us use std::string for text.
#include <string>

// vector lets us store a list-like group of values.
#include <vector>

// This function clamps a value into a safe range.
// double means a decimal number.
double clamp(double value, double minimum, double maximum) {
    // If value is too small, return the minimum allowed value.
    if (value < minimum) {
        return minimum;
    }

    // If value is too large, return the maximum allowed value.
    if (value > maximum) {
        return maximum;
    }

    // Otherwise, return the original value.
    return value;
}

// A class is a blueprint for objects.
class Motor {
public:
    // int stores a whole number.
    int can_id;

    // std::string stores text.
    std::string name;

    // bool stores true or false.
    bool enabled;

    // This is a constructor.
    // It is similar to Python's __init__ method.
    Motor(int input_can_id, std::string input_name) {
        // Store the input CAN ID in this object's can_id.
        can_id = input_can_id;

        // Store the input name in this object's name.
        name = input_name;

        // Start every motor as enabled.
        enabled = true;
    }

    // This method prints information about the motor.
    void describe() {
        // std::cout prints to the terminal.
        std::cout << name << " has CAN ID " << can_id << "\n";
    }
};

// main is where a simple C++ program starts running.
int main() {
    // Print a title.
    std::cout << "C++ Starter: Robot Motor Example\n\n";

    // Create Motor objects.
    Motor hip_motor(3, "hip_front_left");
    Motor thigh_motor(2, "thigh_front_left");
    Motor shank_motor(1, "shank_front_left");

    // Store the motors in a vector.
    std::vector<Motor> motors = {hip_motor, thigh_motor, shank_motor};

    // A for loop repeats once for each motor.
    for (Motor motor : motors) {
        // Call the describe method on the current motor.
        motor.describe();
    }

    // Create a raw motor command.
    double raw_command = 1.8;

    // Clamp the command so it stays between -1.0 and 1.0.
    double safe_command = clamp(raw_command, -1.0, 1.0);

    // Print the raw and safe commands.
    std::cout << "\nRaw command: " << raw_command << "\n";
    std::cout << "Safe command: " << safe_command << "\n\n";

    // Search for a motor by CAN ID.
    int searched_can_id = 2;

    // found starts false because we have not found the motor yet.
    bool found = false;

    // Loop through every motor.
    for (Motor motor : motors) {
        // If this motor has the searched CAN ID, print it.
        if (motor.can_id == searched_can_id) {
            std::cout << "Found motor: " << motor.name << "\n";

            // Mark that we found it.
            found = true;

            // break exits the loop early.
            break;
        }
    }

    // If found is still false, print an error message.
    if (!found) {
        std::cout << "No motor found.\n";
    }

    // A while loop repeats while the condition is true.
    int tick = 0;

    // This simulates three control-loop ticks.
    while (tick < 3) {
        std::cout << "Control tick: " << tick << "\n";

        // Increase tick so the loop eventually stops.
        tick = tick + 1;
    }

    // return 0 tells the operating system the program finished successfully.
    return 0;
}
