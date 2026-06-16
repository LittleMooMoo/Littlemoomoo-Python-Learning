"""Lesson 1 demo: Python basics through motor command safety."""


def clamp_motor_command(command):
    """Limit a motor command to the safe range [-1.0, 1.0]."""
    if command > 1.0:
        return 1.0
    elif command < -1.0:
        return -1.0
    else:
        return command


def describe_command(command):
    """Return a short message describing whether a command was safe."""
    safe_command = clamp_motor_command(command)

    if safe_command == command:
        return "safe"
    else:
        return "clamped"


test_commands = [-1.4, -1.0, -0.25, 0.0, 0.75, 1.0, 1.8]

print("Lesson 1: Motor Command Safety")
print("--------------------------------")

for command in test_commands:
    safe_command = clamp_motor_command(command)
    status = describe_command(command)

    print(f"input: {command:>5} -> output: {safe_command:>4} ({status})")
