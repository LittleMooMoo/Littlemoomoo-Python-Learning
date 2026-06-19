raw_commands = [-2.0, -0.5, 0.3, 2.5]


def clamp(min_value, max_value, value):
    if value < min_value:
        return min_value
    if max_value < value:
        return max_value
    return value


safe_commands = [clamp(-1, 1, raw_command) for raw_command in raw_commands]

print(safe_commands)
