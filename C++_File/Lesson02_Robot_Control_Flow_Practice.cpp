// Lesson02_Robot_Control_Flow_Practice
//
// Goal:
// Practice C++ control flow and helper functions with robotics-flavored problems.
// This file compiles now, but several functions intentionally contain TODO logic.
//
// Suggested workflow:
// 1. Compile and run once.
// 2. Solve Problem 1 only.
// 3. Compile and run again.
// 4. Continue one problem at a time.
//
// Compile with g++:
// g++ -std=c++17 -Wall -Wextra "C++_File/Lesson02_Robot_Control_Flow_Practice.cpp" -o "C++_File/Lesson02_Robot_Control_Flow_Practice.exe"

#include <cmath>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

constexpr double PI = 3.141592653589793;

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

// Solved example:
// Clamp a command so it never goes above +limit or below -limit.
double clamp_symmetric(double value, double limit)
{
    if (limit < 0.0)
    {
        limit = -limit;
    }

    if (value > limit)
    {
        return limit;
    }
    else if (value < -limit)
    {
        return -limit;
    }
    else
    {
        return value;
    }
}

// Problem 1:
// Return a readable string for every MotorMode.
//
// Hint:
// Use if / else if / else.
//
// Expected:
// MotorMode::Idle     -> "idle"
// MotorMode::Position -> "position"
// MotorMode::Torque   -> "torque"
// MotorMode::Error    -> "error"
std::string mode_to_string(MotorMode mode)
{
    if (mode == MotorMode::Idle)
    {
        return "idle";
    }

    return "TODO_mode";
}

// Problem 2:
// Check if a motor target is safe.
//
// Rules:
// - Offline motor is never safe.
// - Motor in Error mode is never safe.
// - Target must be between lower_limit and upper_limit.
//
// Hint:
// Use nested if or several if statements with early returns.
bool is_target_safe(const MotorState& motor, double lower_limit, double upper_limit)
{
    (void)motor;
    (void)lower_limit;
    (void)upper_limit;

    return false;
}

// Problem 3:
// Compute a PD torque command:
//
// position_error = target_position - measured_position
// velocity_error = target_velocity - measured_velocity
// torque = kp * position_error + kd * velocity_error
//
// Then clamp the torque using clamp_symmetric().
double calculate_pd_torque(
    double target_position,
    double measured_position,
    double target_velocity,
    double measured_velocity,
    double kp,
    double kd,
    double torque_limit)
{
    (void)target_position;
    (void)measured_position;
    (void)target_velocity;
    (void)measured_velocity;
    (void)kp;
    (void)kd;
    (void)torque_limit;

    return 0.0;
}

// Problem 4:
// Print only the online motors.
//
// Example output:
// online motor: 1 front_right_shank
// online motor: 2 front_right_thigh
//
// Hint:
// Use a range-based for loop:
// for (const MotorState& motor : motors)
void print_online_motors(const std::vector<MotorState>& motors)
{
    (void)motors;

    std::cout << "TODO: print online motors\n";
}

// Problem 5:
// Find the index of a motor by CAN id.
//
// Return:
// - index if found
// - -1 if not found
//
// Hint:
// Use a normal for loop because you need the index.
int find_motor_index_by_id(const std::vector<MotorState>& motors, int wanted_id)
{
    (void)motors;
    (void)wanted_id;

    return -1;
}

// Problem 6:
// Set all online motors to a new mode.
//
// Important C++ idea:
// motors is passed by reference using &, so this function changes the original vector.
//
// Hint:
// Use:
// for (MotorState& motor : motors)
void set_online_motor_mode(std::vector<MotorState>& motors, MotorMode new_mode)
{
    (void)motors;
    (void)new_mode;
}

// Problem 7:
// Build a smooth cosine trajectory from start_position to target_position.
//
// Use this formula:
// phase = i / steps
// blend = 0.5 * (1.0 - cos(PI * phase))
// command = start_position + (target_position - start_position) * blend
//
// If steps = 5, return 6 values:
// i = 0, 1, 2, 3, 4, 5
//
// Hint:
// Use push_back() to append values to the vector.
std::vector<double> build_cosine_trajectory(
    double start_position,
    double target_position,
    int steps)
{
    (void)start_position;
    (void)target_position;
    (void)steps;

    std::vector<double> trajectory;
    return trajectory;
}

// Problem 8:
// Ask the user whether to keep a fake robot online.
//
// Rules:
// - 'y' or 'Y': print "staying online"
// - 'n' or 'N': print "going offline" and stop the loop
// - anything else: print "invalid input" and ask again
//
// Hint:
// Use while (is_online) and std::cin >> response.
void fake_online_loop()
{
    bool is_online = true;
    char response = '\0';

    std::cout << "\nProblem 8 interactive practice.\n";
    std::cout << "Type n to skip/quit this fake online loop.\n";

    while (is_online)
    {
        std::cout << "Keep fake robot online? y/n: ";
        std::cin >> response;

        if (response == 'n' || response == 'N')
        {
            std::cout << "going offline\n";
            is_online = false;
        }
        else
        {
            std::cout << "TODO: handle y/Y and invalid input better\n";
            is_online = false;
        }
    }
}

void print_motor_summary(const MotorState& motor)
{
    std::cout << "motor " << motor.id
              << " | " << motor.name
              << " | mode: " << mode_to_string(motor.mode)
              << " | measured: " << motor.measured_position
              << " | target: " << motor.target_position
              << " | online: " << std::boolalpha << motor.online
              << "\n";
}

int main()
{
    std::vector<MotorState> motors = {
        {1, "front_right_shank", 0.10, 0.00, 0.40, true, MotorMode::Position},
        {2, "front_right_thigh", -0.20, 0.05, -0.60, true, MotorMode::Idle},
        {3, "front_right_hip", 0.00, 0.00, 1.50, false, MotorMode::Position},
        {4, "front_left_shank", 0.30, -0.02, 0.20, true, MotorMode::Error}
    };

    std::cout << "=== Robot C++ Control Flow Practice ===\n\n";

    std::cout << "Initial motor table:\n";
    for (const MotorState& motor : motors)
    {
        print_motor_summary(motor);
    }

    std::cout << "\nProblem 1 check:\n";
    std::cout << "Motor 1 mode should eventually print 'position': "
              << mode_to_string(motors[0].mode) << "\n";

    std::cout << "\nProblem 2 check:\n";
    std::cout << "Motor 1 target safe between -1.0 and 1.0? "
              << std::boolalpha << is_target_safe(motors[0], -1.0, 1.0) << "\n";
    std::cout << "Motor 3 target safe between -1.0 and 1.0? "
              << std::boolalpha << is_target_safe(motors[2], -1.0, 1.0) << "\n";

    std::cout << "\nProblem 3 check:\n";
    double torque = calculate_pd_torque(
        motors[0].target_position,
        motors[0].measured_position,
        0.0,
        motors[0].measured_velocity,
        0.8,
        0.02,
        0.3);
    std::cout << "PD torque command: " << torque << "\n";

    std::cout << "\nProblem 4 check:\n";
    print_online_motors(motors);

    std::cout << "\nProblem 5 check:\n";
    int index = find_motor_index_by_id(motors, 2);
    std::cout << "Index of motor id 2: " << index << "\n";

    std::cout << "\nProblem 6 check:\n";
    set_online_motor_mode(motors, MotorMode::Position);
    for (const MotorState& motor : motors)
    {
        print_motor_summary(motor);
    }

    std::cout << "\nProblem 7 check:\n";
    std::vector<double> trajectory = build_cosine_trajectory(0.0, 1.0, 5);
    std::cout << "Trajectory values:\n";
    for (double point : trajectory)
    {
        std::cout << std::fixed << std::setprecision(3) << point << "\n";
    }

    fake_online_loop();

    return 0;
}
