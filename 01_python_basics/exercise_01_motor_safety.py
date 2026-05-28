"""Lesson 1 exercise: edit the safety limits and predict the output."""

MAX_COMMAND = 1.0
MIN_COMMAND = -1.0


def clamp_motor_command(command):
    """Limit a motor command to the configured safe range."""
    if command > MAX_COMMAND:
        return MAX_COMMAND
    elif command < MIN_COMMAND:
        return MIN_COMMAND
    else:
        return command


test_commands = [-1.4, -0.8, -0.2, 0.0, 0.4, 0.9, 1.6]

print("Exercise 1: Change the Safety Limits")
print("------------------------------------")
print(f"allowed range: {MIN_COMMAND} to {MAX_COMMAND}")

for command in test_commands:
    safe_command = clamp_motor_command(command)
    print(f"input: {command:>5} -> output: {safe_command:>4}")
