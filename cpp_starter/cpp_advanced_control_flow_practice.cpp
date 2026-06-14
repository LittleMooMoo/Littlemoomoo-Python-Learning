// C++ advanced beginner lesson: control flow, errors, and practice.
//
// This file records ideas beyond basic variables/functions/classes:
//
// - while loops
// - break and continue
// - vectors
// - maps
// - try / catch
// - safer parsing
// - structs
// - helper functions
// - robotics-style command safety
// - practice problems written as comments
//
// Compile later with:
//
//     New-Item -ItemType Directory -Force build
//     g++ cpp_starter\cpp_advanced_control_flow_practice.cpp -std=c++17 -Wall -Wextra -o build\cpp_advanced_control_flow_practice.exe
//     .\build\cpp_advanced_control_flow_practice.exe

#include <iostream>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

double clamp(double value, double minimum, double maximum) {
    if (value < minimum) {
        return minimum;
    }

    if (value > maximum) {
        return maximum;
    }

    return value;
}

bool try_parse_double(const std::string& text, double& output_value) {
    try {
        output_value = std::stod(text);
        return true;
    } catch (const std::invalid_argument&) {
        return false;
    } catch (const std::out_of_range&) {
        return false;
    }
}

struct Motor {
    int can_id;
    std::string name;
    bool enabled;
};

void print_motor(const Motor& motor) {
    std::cout << motor.name << " | CAN ID: " << motor.can_id
              << " | enabled: " << motor.enabled << "\n";
}

int main() {
    std::cout << "C++ Advanced Control Flow Practice\n\n";

    std::cout << "--- while loop ---\n";

    int control_tick = 0;

    while (control_tick < 3) {
        std::cout << "control tick: " << control_tick << "\n";
        control_tick = control_tick + 1;
    }

    std::cout << "\n--- break and continue ---\n";

    std::vector<int> motor_ids = {1, 2, 3, 4};

    for (int motor_id : motor_ids) {
        if (motor_id == 2) {
            std::cout << "skip disabled motor: " << motor_id << "\n";
            continue;
        }

        if (motor_id == 4) {
            std::cout << "emergency stop at motor: " << motor_id << "\n";
            break;
        }

        std::cout << "update motor: " << motor_id << "\n";
    }

    std::cout << "\n--- map lookup ---\n";

    std::map<std::string, int> motor_name_to_can_id = {
        {"hip_front_left", 3},
        {"thigh_front_left", 2},
        {"shank_front_left", 1},
    };

    std::string searched_name = "hip_front_left";

    if (motor_name_to_can_id.count(searched_name) > 0) {
        std::cout << searched_name << " CAN ID: "
                  << motor_name_to_can_id[searched_name] << "\n";
    } else {
        std::cout << "motor not found\n";
    }

    std::cout << "\n--- try / catch parsing ---\n";

    std::vector<std::string> example_inputs = {"0.5", "1.8", "-2.2", "bad_input"};

    for (std::string text : example_inputs) {
        double parsed_command = 0.0;
        bool parsed_ok = try_parse_double(text, parsed_command);

        if (!parsed_ok) {
            std::cout << text << " -> invalid input\n";
            continue;
        }

        double safe_command = clamp(parsed_command, -1.0, 1.0);
        std::cout << text << " -> " << safe_command << "\n";
    }

    std::cout << "\n--- struct list ---\n";

    std::vector<Motor> motors = {
        {3, "hip_front_left", true},
        {2, "thigh_front_left", false},
        {1, "shank_front_left", true},
    };

    for (const Motor& motor : motors) {
        if (!motor.enabled) {
            std::cout << "skip disabled motor: " << motor.name << "\n";
            continue;
        }

        print_motor(motor);
    }

    return 0;
}

// ---------------------------------------------------------------------------
// PRACTICE PROBLEMS
// ---------------------------------------------------------------------------

// Practice 1:
// Write a function named is_safe_angle.
// It should take a double angle_deg.
// It should return true if the angle is between -45 and 45.
// Otherwise, return false.


// Practice 2:
// Write a while loop that asks the user for a motor command.
// If the user types q, stop the loop.
// If the user types a number, clamp it between -1.0 and 1.0 and print it.
// If the user types invalid text, print "invalid command".


// Practice 3:
// Create a std::map that maps motor names to CAN IDs:
//
//     hip_front_left -> 3
//     thigh_front_left -> 2
//     shank_front_left -> 1
//
// Ask the user for a motor name and print the CAN ID.
// If the motor is not found, print "motor not found".


// Practice 4:
// Create a vector of raw commands:
//
//     {-2.0, -0.5, 0.3, 2.5}
//
// Use a for loop to create a second vector of safe commands.


// Practice 5:
// Create a struct named JointLimit.
// It should store:
//
//     std::string joint_name
//     double min_angle
//     double max_angle
//
// Create one JointLimit for a hip joint and print its values.


// Practice 6:
// Create a vector of Motor structs.
// Loop through them.
// Use continue to skip disabled motors.
// Print only enabled motors.
