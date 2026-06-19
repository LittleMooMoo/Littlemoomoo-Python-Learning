
commands = [-2.0, -0.5, 0.3, 2.5]


def clamp(min_value, max_value, value):
    if max_value < value:
        return max_value
    if value < min_value:
        return min_value
    return value


safe_commands = []
for raw_commands in commands:
    safe_command = clamp(-1.0, 1.0, raw_commands)
    safe_commands.append(safe_command)
print(safe_commands)
