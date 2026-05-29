r"""Python concept map for beginner robotics programming.

This file is meant to be read slowly.

Almost every line has a comment explaining what it does.
The code that is not commented out will run from top to bottom.
The code that starts with # is a note for you, not code Python will execute.

Run from the repo root with:

    & "$env:LOCALAPPDATA\Programs\Python\Python313\python.exe" 01_python_basics\python_concept_map.py
"""

# ---------------------------------------------------------------------------
# 1. IMPORTS
# ---------------------------------------------------------------------------

# import means: bring in code from another Python module or library.
# math is a built-in Python library for math functions like sin, cos, sqrt.
import math

# dataclass is a helper that can create simple data-storage classes.
from dataclasses import dataclass


# ---------------------------------------------------------------------------
# 2. PRINT
# ---------------------------------------------------------------------------

# print shows text or values in the terminal.
print("Python Concept Map: Robotics Edition")

# This empty print creates a blank line so the output is easier to read.
print()


# ---------------------------------------------------------------------------
# 3. VARIABLES
# ---------------------------------------------------------------------------

# A variable is a name that stores a value.
robot_name = "LittleMooMoo Robot"

# This variable stores a whole number, also called an int.
number_of_legs = 4

# This variable stores a decimal number, also called a float.
battery_voltage = 14.8

# This variable stores True or False, also called a bool.
robot_enabled = True

# We can print variables to see what they store.
print("Robot name:", robot_name)
print("Number of legs:", number_of_legs)
print("Battery voltage:", battery_voltage)
print("Robot enabled:", robot_enabled)
print()


# ---------------------------------------------------------------------------
# 4. BASIC DATA TYPES
# ---------------------------------------------------------------------------

# int means integer, which is a whole number.
motor_id = 3

# float means floating-point number, which is a decimal number.
hip_angle_degrees = 32.5

# str means string, which is text.
joint_name = "front_left_hip"

# bool means boolean, which is True or False.
is_joint_safe = True

# type(...) tells you what type a value has.
print("motor_id type:", type(motor_id))
print("hip_angle_degrees type:", type(hip_angle_degrees))
print("joint_name type:", type(joint_name))
print("is_joint_safe type:", type(is_joint_safe))
print()


# ---------------------------------------------------------------------------
# 5. STRINGS
# ---------------------------------------------------------------------------

# A string stores text.
leg_name = "front_left"

# You can combine strings with +.
full_joint_name = leg_name + "_hip"

# An f-string lets you put variables directly inside text.
debug_message = f"Joint {full_joint_name} is at {hip_angle_degrees} degrees."

# Print the finished message.
print(debug_message)
print()


# ---------------------------------------------------------------------------
# 6. IF, ELIF, ELSE
# ---------------------------------------------------------------------------

# if means: only run this block if the condition is True.
# elif means: otherwise, check this next condition.
# else means: if nothing above was True, run this block.

motor_command = 1.25

if motor_command > 1.0:
    print("Motor command is too high.")
elif motor_command < -1.0:
    print("Motor command is too low.")
else:
    print("Motor command is safe.")

print()


# ---------------------------------------------------------------------------
# 7. FUNCTIONS, ARGUMENTS, AND RETURN
# ---------------------------------------------------------------------------

# def means: define a function.
# command is an argument, which means an input to the function.
def clamp_motor_command(command):
    # If command is bigger than the safe max, return the max.
    if command > 1.0:
        return 1.0

    # If command is smaller than the safe min, return the min.
    if command < -1.0:
        return -1.0

    # If command is already safe, return it unchanged.
    return command


# Here we call the function with 1.25 as the argument.
safe_command = clamp_motor_command(1.25)

# The return value from the function is stored in safe_command.
print("Clamped motor command:", safe_command)
print()


# A function can have more than one argument.
def calculate_joint_error(target_angle, measured_angle):
    # Error means how far measured_angle is away from target_angle.
    error = target_angle - measured_angle

    # Return sends the answer back to the code that called the function.
    return error


# These are example input values.
target_hip_angle = 30.0
measured_hip_angle = 24.5

# We call the function and store its returned value.
hip_error = calculate_joint_error(target_hip_angle, measured_hip_angle)

# Print the result.
print("Hip angle error:", hip_error)
print()


# ---------------------------------------------------------------------------
# 8. LISTS AND FOR LOOPS
# ---------------------------------------------------------------------------

# A list stores multiple values in one variable.
motor_commands = [-1.4, -0.5, 0.25, 1.8]

# A for loop repeats code for every item in a list.
for command in motor_commands:
    # command is the current item from the list.
    safe_command = clamp_motor_command(command)

    # Print both the original command and the safe command.
    print("raw command:", command, "safe command:", safe_command)

print()


# ---------------------------------------------------------------------------
# 9. DICTIONARIES
# ---------------------------------------------------------------------------

# A dictionary stores key-value pairs.
# Here, the key is the motor name, and the value is the CAN ID.
motor_name_to_can_id = {
    "hip_front_left": 3,
    "thigh_front_left": 2,
    "shank_front_left": 1,
}

# We can look up a value using its key.
front_left_hip_id = motor_name_to_can_id["hip_front_left"]

# Print the result of the lookup.
print("CAN ID for hip_front_left:", front_left_hip_id)

# .get(...) is safer than direct lookup because it returns None if not found.
unknown_motor_id = motor_name_to_can_id.get("unknown_motor")

# None means no value was found.
print("CAN ID for unknown_motor:", unknown_motor_id)
print()


# ---------------------------------------------------------------------------
# 10. CLASSES, OBJECTS, ATTRIBUTES, AND METHODS
# ---------------------------------------------------------------------------

# A class is a blueprint.
# This blueprint describes what information an actuator config should store.
class ActuatorConfig:
    # __init__ runs automatically when a new object is created.
    def __init__(self, can_id, motor_name, joint_name):
        # self.can_id stores the CAN ID inside this specific object.
        self.can_id = can_id

        # self.motor_name stores the motor name inside this object.
        self.motor_name = motor_name

        # self.joint_name stores which joint this actuator controls.
        self.joint_name = joint_name

    # A function inside a class is called a method.
    def describe(self):
        # This method returns a readable sentence about this actuator.
        return f"{self.motor_name}: CAN ID {self.can_id}, joint {self.joint_name}"


# Create actual objects from the ActuatorConfig class.
hip_front_left = ActuatorConfig(3, "hip_front_left", "hip")
thigh_front_left = ActuatorConfig(2, "thigh_front_left", "thigh")
shank_front_left = ActuatorConfig(1, "shank_front_left", "shank")

# Store the objects in a list.
actuators = [hip_front_left, thigh_front_left, shank_front_left]

# Loop through the actuator objects.
for actuator in actuators:
    # Call the describe method on the current actuator.
    print(actuator.describe())

print()


# ---------------------------------------------------------------------------
# 11. SEARCHING A LIST OF OBJECTS
# ---------------------------------------------------------------------------

# This function searches the actuator list by CAN ID.
def find_actuator_by_can_id(actuator_list, searched_can_id):
    # None means we have not found a matching actuator yet.
    found_actuator = None

    # Loop through every actuator in the list.
    for actuator in actuator_list:
        # Check whether the current actuator has the CAN ID we want.
        if actuator.can_id == searched_can_id:
            # Store the matching actuator object.
            found_actuator = actuator

            # Stop the loop because we already found the answer.
            break

    # Return either the matching actuator object or None.
    return found_actuator


# Search for CAN ID 2.
searched_actuator = find_actuator_by_can_id(actuators, 2)

# Check whether the search found something.
if searched_actuator is None:
    print("No actuator found.")
else:
    print("Found actuator:", searched_actuator.describe())

print()


# ---------------------------------------------------------------------------
# 12. DATACLASSES
# ---------------------------------------------------------------------------

# @dataclass tells Python to create a simple __init__ automatically.
# This is useful for robot configuration data.
@dataclass
class JointLimit:
    # min_angle stores the lowest allowed joint angle.
    min_angle: float

    # max_angle stores the highest allowed joint angle.
    max_angle: float


# Create a JointLimit object.
hip_limit = JointLimit(min_angle=-45.0, max_angle=45.0)

# Print its stored values.
print("Hip min angle:", hip_limit.min_angle)
print("Hip max angle:", hip_limit.max_angle)
print()


# ---------------------------------------------------------------------------
# 13. IMPORTED MATH FUNCTIONS
# ---------------------------------------------------------------------------

# math.radians converts degrees to radians.
angle_radians = math.radians(30.0)

# math.sin calculates the sine of an angle in radians.
sin_value = math.sin(angle_radians)

# Print the result.
print("sin(30 degrees):", sin_value)

# math.sqrt calculates a square root.
distance = math.sqrt(3.0**2 + 4.0**2)

# This is the Pythagorean theorem: sqrt(x^2 + y^2).
print("Distance from x=3, y=4:", distance)
print()


# ---------------------------------------------------------------------------
# 14. ROBOTICS MINI EXAMPLE: SIMPLE POSITION CONTROL
# ---------------------------------------------------------------------------

# This is a very small proportional controller.
# It is not a complete real robot controller, but it teaches the idea.
def proportional_control(target_angle, measured_angle, gain):
    # Calculate how far the joint is from the target.
    error = target_angle - measured_angle

    # Multiply the error by a gain to create a command.
    raw_command = gain * error

    # Clamp the command so it stays in the safe range.
    safe_command = clamp_motor_command(raw_command)

    # Return the safe command.
    return safe_command


# Example target and measured angle.
target_angle = 20.0
measured_angle = 15.0

# Gain controls how strongly the controller reacts to error.
kp = 0.1

# Calculate the motor command.
controller_output = proportional_control(target_angle, measured_angle, kp)

# Print the command.
print("Controller output:", controller_output)
print()


# ---------------------------------------------------------------------------
# 15. VIRTUAL ENVIRONMENT NOTES
# ---------------------------------------------------------------------------

# A virtual environment is an isolated Python workspace for one project.
# It helps keep one project's packages separate from another project's packages.
#
# You usually create one from the terminal, not from inside a Python script.
#
# On Windows PowerShell, from the repo root:
#
#     & "$env:LOCALAPPDATA\Programs\Python\Python313\python.exe" -m venv .venv
#
# Then activate it:
#
#     .\.venv\Scripts\Activate.ps1
#
# After activation, installing packages affects this project environment:
#
#     python -m pip install ruff
#
# ruff is a linter and formatter. A linter checks code style and likely bugs.
#
# You do not need to master virtual environments today.
# For now, remember:
#
#     venv = project-specific Python environment
#     pip = installs packages
#     linter = checks code quality

print("Virtual environment notes are in the comments near the bottom of this file.")
print()


# ---------------------------------------------------------------------------
# 16. OPTIONAL INPUT SECTION
# ---------------------------------------------------------------------------

# This section is commented out so the file runs without waiting for input.
# Later, you can remove the # marks to practice terminal input.
#
# user_text = input("Type a CAN ID or motor name: ")
#
# if user_text.isdigit():
#     user_can_id = int(user_text)
#     result = find_actuator_by_can_id(actuators, user_can_id)
#     if result is None:
#         print("No actuator found.")
#     else:
#         print(result.describe())
# else:
#     print("You typed a motor name:", user_text)


# ---------------------------------------------------------------------------
# 17. MAIN IDEA SUMMARY
# ---------------------------------------------------------------------------

print("Summary:")
print("- variables store values")
print("- functions perform reusable actions")
print("- arguments are inputs to functions")
print("- return sends a value back")
print("- classes are blueprints")
print("- objects are actual things made from classes")
print("- imports bring in useful outside code")
print("- lists and dictionaries store groups of values")
print("- robot code uses all of these ideas together")
