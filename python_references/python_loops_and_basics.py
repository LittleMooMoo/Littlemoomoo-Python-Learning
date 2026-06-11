r"""Python loops and basics reference for robotics beginners.

Run from the repo root:

    & "$env:LOCALAPPDATA\Programs\Python\Python313\python.exe" python_references\python_loops_and_basics.py

This file is intentionally full of comments.
Comments start with # and are ignored by Python.
"""

# A variable stores a value.
robot_name = "LittleMooMoo Robot"

# An int stores a whole number.
number_of_motors = 3

# A float stores a decimal number.
max_motor_command = 1.0

# A bool stores True or False.
robot_enabled = True

# print sends information to the terminal.
print("Robot name:", robot_name)
print("Number of motors:", number_of_motors)
print("Max motor command:", max_motor_command)
print("Robot enabled:", robot_enabled)
print()


# def creates a function.
# value, minimum, and maximum are arguments, which are inputs to the function.
def clamp(value, minimum, maximum):
    # If value is below the minimum, return the minimum.
    if value < minimum:
        return minimum

    # If value is above the maximum, return the maximum.
    if value > maximum:
        return maximum

    # Otherwise, value is already safe, so return it unchanged.
    return value


# A list stores multiple values.
raw_motor_commands = [-1.5, -0.2, 0.4, 1.8]

# A for loop repeats once for each item in a list.
for command in raw_motor_commands:
    # Clamp each command so it stays between -1.0 and 1.0.
    safe_command = clamp(command, -1.0, 1.0)

    # Print the raw command and the safe command.
    print("raw:", command, "safe:", safe_command)

print()


# A class is a blueprint for objects.
class Motor:
    # __init__ runs when a new Motor object is created.
    def __init__(self, can_id, name):
        # self.can_id stores this motor's CAN ID.
        self.can_id = can_id

        # self.name stores this motor's name.
        self.name = name

        # self.enabled stores whether this motor is allowed to move.
        self.enabled = True

    # A function inside a class is called a method.
    def describe(self):
        # Return a readable description of this motor.
        return f"{self.name} has CAN ID {self.can_id}"


# Create three Motor objects.
hip_motor = Motor(3, "hip_front_left")
thigh_motor = Motor(2, "thigh_front_left")
shank_motor = Motor(1, "shank_front_left")

# Store the motors in a list.
motors = [hip_motor, thigh_motor, shank_motor]

# Search for a motor by CAN ID.
searched_can_id = 2

# None means no motor has been found yet.
found_motor = None

# Loop through every motor.
for motor in motors:
    # If this motor has the CAN ID we want, store it.
    if motor.can_id == searched_can_id:
        found_motor = motor

        # break exits the loop because we found the answer.
        break

# Check whether the search worked.
if found_motor is None:
    print("No motor found.")
else:
    print("Found motor:", found_motor.describe())

print()


# continue skips the rest of the current loop and moves to the next item.
for motor in motors:
    # Pretend the thigh motor is disabled.
    if motor.name == "thigh_front_left":
        motor.enabled = False

    # If the motor is disabled, skip it.
    if not motor.enabled:
        print("Skipping disabled motor:", motor.name)
        continue

    # This only runs for enabled motors.
    print("Updating enabled motor:", motor.name)

print()


# A while loop repeats as long as a condition is true.
counter = 0

# This loop runs while counter is less than 3.
while counter < 3:
    print("Control loop tick:", counter)

    # Increase counter so the loop eventually stops.
    counter = counter + 1

print()


# A dictionary stores key-value pairs.
motor_name_to_id = {
    "hip_front_left": 3,
    "thigh_front_left": 2,
    "shank_front_left": 1,
}

# Look up a CAN ID by motor name.
print("hip_front_left ID:", motor_name_to_id["hip_front_left"])

# .get returns None if the key is not found.
print("unknown motor ID:", motor_name_to_id.get("unknown_motor"))
