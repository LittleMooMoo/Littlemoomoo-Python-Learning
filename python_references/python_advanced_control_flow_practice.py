r"""Python advanced beginner lesson: control flow, errors, and practice.

Run from the repo root:

    & "$env:LOCALAPPDATA\Programs\Python\Python313\python.exe" python_references\python_advanced_control_flow_practice.py

This file is still beginner-friendly, but it introduces more serious habits:

- try / except for handling errors
- while loops
- break and continue
- dictionaries
- list comprehensions
- helper functions with one clear job
- small robotics-style safety checks
- practice problems written as comments
"""


# ---------------------------------------------------------------------------
# 1. TRY / EXCEPT
# ---------------------------------------------------------------------------

# try / except lets your program handle an error instead of crashing.
# This is extremely important in robotics because sensors, user input, files,
# and communication can all fail.


def text_to_float(text):
    """Convert text to a float, or return None if conversion fails."""
    try:
        # Try to convert the text into a decimal number.
        value = float(text)

        # If conversion worked, return the number.
        return value

    except ValueError:
        # ValueError happens when text cannot become a float.
        return None


print("--- try / except example ---")
print(text_to_float("3.14"))
print(text_to_float("moo"))
print()


# ---------------------------------------------------------------------------
# 2. WHILE LOOP
# ---------------------------------------------------------------------------

# A while loop repeats while a condition is true.
# Be careful: if the condition never becomes false, the loop runs forever.

print("--- while loop example ---")

control_tick = 0

while control_tick < 3:
    print("control tick:", control_tick)
    control_tick = control_tick + 1

print()


# ---------------------------------------------------------------------------
# 3. BREAK AND CONTINUE
# ---------------------------------------------------------------------------

# break stops the loop immediately.
# continue skips the current loop step and moves to the next one.

print("--- break and continue example ---")

motor_ids = [1, 2, 3, 4]

for motor_id in motor_ids:
    if motor_id == 2:
        print("skip disabled motor:", motor_id)
        continue

    if motor_id == 4:
        print("emergency stop at motor:", motor_id)
        break

    print("update motor:", motor_id)

print()


# ---------------------------------------------------------------------------
# 4. DICTIONARIES
# ---------------------------------------------------------------------------

# A dictionary maps a key to a value.
# This is useful for motor name -> CAN ID lookup.

motor_name_to_can_id = {
    "hip_front_left": 3,
    "thigh_front_left": 2,
    "shank_front_left": 1,
}

print("--- dictionary example ---")
print("hip_front_left ID:", motor_name_to_can_id["hip_front_left"])
print("unknown motor ID:", motor_name_to_can_id.get("unknown_motor"))
print()


# ---------------------------------------------------------------------------
# 5. LIST COMPREHENSION
# ---------------------------------------------------------------------------

# A list comprehension is a compact way to build a new list.

raw_commands = [-1.5, -0.2, 0.4, 1.8]


def clamp(value, minimum, maximum):
    """Limit value so it stays inside [minimum, maximum]."""
    if value < minimum:
        return minimum
    if value > maximum:
        return maximum
    return value


safe_commands = [clamp(command, -1.0, 1.0) for command in raw_commands]

print("--- list comprehension example ---")
print("raw commands:", raw_commands)
print("safe commands:", safe_commands)
print()


# ---------------------------------------------------------------------------
# 6. ROBOTICS MINI EXAMPLE: SAFE COMMAND PIPELINE
# ---------------------------------------------------------------------------

# A pipeline means a sequence of steps:
#
#     raw input -> parse -> validate -> clamp -> output
#
# This pattern appears constantly in robot software.


def parse_motor_command(text):
    """Parse a motor command from text and return a safe command or None."""
    command = text_to_float(text)

    if command is None:
        return None

    return clamp(command, -1.0, 1.0)


print("--- safe command pipeline ---")

example_inputs = ["0.5", "1.8", "-2.2", "bad_input"]

for text in example_inputs:
    safe_command = parse_motor_command(text)

    if safe_command is None:
        print(text, "-> invalid input")
    else:
        print(text, "->", safe_command)

print()


# ---------------------------------------------------------------------------
# 7. NESTED DATA STRUCTURES
# ---------------------------------------------------------------------------

# A list can contain dictionaries.
# Each dictionary below describes one motor.

motors = [
    {"can_id": 3, "name": "hip_front_left", "enabled": True},
    {"can_id": 2, "name": "thigh_front_left", "enabled": False},
    {"can_id": 1, "name": "shank_front_left", "enabled": True},
]

print("--- nested data structure example ---")

for motor in motors:
    if not motor["enabled"]:
        print("skip disabled motor:", motor["name"])
        continue

    print("ready motor:", motor["name"], "CAN ID:", motor["can_id"])

print()


# ---------------------------------------------------------------------------
# 8. PRACTICE PROBLEMS
# ---------------------------------------------------------------------------

# Practice 1:
# Write a function named is_safe_angle(angle_deg).
# It should return True if angle_deg is between -45 and 45.
# Otherwise, return False.


# Practice 2:
# Write a while loop that asks the user for a motor command.
# If the user types q, stop the loop.
# If the user types a number, clamp it between -1.0 and 1.0 and print it.
# If the user types invalid text, print "invalid command".


# Practice 3:
# Create a dictionary that maps these motor names to CAN IDs:
#
#     hip_front_left -> 3
#     thigh_front_left -> 2
#     shank_front_left -> 1
#
# Ask the user for a motor name and print the CAN ID.
# If the motor is not found, print "motor not found".


# Practice 4:
# Given a list of raw commands:
#
#     commands = [-2.0, -0.5, 0.3, 2.5]
#
# Use a for loop to create a new list of safe commands.
# Do not use list comprehension yet.


# Practice 5:
# Now solve Practice 4 again using a list comprehension.


# Practice 6:
# Create a list of motor dictionaries.
# Each motor should have:
#
#     can_id
#     name
#     enabled
#
# Loop through the motors.
# Use continue to skip disabled motors.
# Print only enabled motors.
