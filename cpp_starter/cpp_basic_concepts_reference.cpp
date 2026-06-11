// C++ Basic Concepts Reference
//
// This file records the beginner C++ ideas you asked about.
// It is written as one runnable file with lots of comments.
//
// Compile later with:
//
//     New-Item -ItemType Directory -Force build
//     g++ cpp_starter\cpp_basic_concepts_reference.cpp -std=c++17 -Wall -Wextra -o build\cpp_basic_concepts_reference.exe
//     .\build\cpp_basic_concepts_reference.exe

// #include is like Python import.
// It brings in code from the C++ standard library.
#include <iostream>  // std::cout and std::cin
#include <string>    // std::string
#include <vector>    // std::vector

// A function is reusable code.
// This function returns a double, which is a decimal number.
double clamp(double value, double minimum, double maximum) {
    if (value < minimum) {
        return minimum;
    } else if (value > maximum) {
        return maximum;
    } else {
        return value;
    }
}

// void means this function does not return a value.
void print_section_title(std::string title) {
    std::cout << "\n--- " << title << " ---\n";
}

// A class is a blueprint for making objects.
class Motor {
public:
    int can_id;
    std::string name;
    double command;

    // This is a constructor.
    // It runs when you create a Motor object.
    Motor(int input_can_id, std::string input_name) {
        can_id = input_can_id;
        name = input_name;
        command = 0.0;
    }

    // This is a method, which means a function inside a class.
    void set_command(double raw_command) {
        command = clamp(raw_command, -1.0, 1.0);
    }

    void describe() {
        std::cout << name << " | CAN ID: " << can_id
                  << " | command: " << command << "\n";
    }
};

// main is where a simple C++ program starts running.
int main() {
    print_section_title("Printing A String");

    // To print a string in C++, use std::cout.
    std::cout << "Hello from C++\n";

    // You can store text in std::string.
    std::string robot_name = "LittleMooMoo Robot";
    std::cout << robot_name << "\n";

    // You can combine strings and variables in one print statement.
    std::cout << "Robot name: " << robot_name << "\n";

    print_section_title("Variables And Types");

    int motor_id = 3;                 // whole number
    double motor_command = 1.8;       // decimal number
    bool motor_enabled = true;        // true or false
    std::string motor_name = "hip_front_left";

    std::cout << "Motor ID: " << motor_id << "\n";
    std::cout << "Motor command: " << motor_command << "\n";
    std::cout << "Motor enabled: " << motor_enabled << "\n";
    std::cout << "Motor name: " << motor_name << "\n";

    print_section_title("If Else Decision");

    if (motor_command > 1.0) {
        std::cout << "Command is too high\n";
    } else if (motor_command < -1.0) {
        std::cout << "Command is too low\n";
    } else {
        std::cout << "Command is safe\n";
    }

    print_section_title("Function And Return");

    double safe_command = clamp(motor_command, -1.0, 1.0);
    std::cout << "Raw command: " << motor_command << "\n";
    std::cout << "Safe command: " << safe_command << "\n";

    print_section_title("Vector And For Loop");

    // std::vector is similar to a Python list.
    std::vector<int> can_ids = {3, 2, 1};

    // This for loop visits every CAN ID in the vector.
    for (int can_id : can_ids) {
        std::cout << "CAN ID: " << can_id << "\n";
    }

    print_section_title("While Loop");

    int tick = 0;

    while (tick < 3) {
        std::cout << "Control tick: " << tick << "\n";
        tick = tick + 1;
    }

    print_section_title("Continue And Break");

    for (int can_id : can_ids) {
        if (can_id == 2) {
            std::cout << "Skipping CAN ID 2\n";
            continue;
        }

        if (can_id == 1) {
            std::cout << "Stopping at CAN ID 1\n";
            break;
        }

        std::cout << "Processing CAN ID: " << can_id << "\n";
    }

    print_section_title("Class And Object");

    Motor hip_motor(3, "hip_front_left");
    Motor thigh_motor(2, "thigh_front_left");
    Motor shank_motor(1, "shank_front_left");

    hip_motor.set_command(0.5);
    thigh_motor.set_command(1.7);
    shank_motor.set_command(-1.4);

    hip_motor.describe();
    thigh_motor.describe();
    shank_motor.describe();

    print_section_title("List Of Objects");

    std::vector<Motor> motors = {hip_motor, thigh_motor, shank_motor};

    int searched_can_id = 2;
    bool found_motor = false;

    for (Motor motor : motors) {
        if (motor.can_id == searched_can_id) {
            std::cout << "Found motor: " << motor.name << "\n";
            found_motor = true;
            break;
        }
    }

    if (!found_motor) {
        std::cout << "No motor found\n";
    }

    return 0;
}
