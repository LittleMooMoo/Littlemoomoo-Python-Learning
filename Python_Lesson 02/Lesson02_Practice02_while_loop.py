def clamp(min_value, max_value, value):
    if value > max_value:
        return max_value
    if value < min_value:
        return min_value
    else:
        return value


def parse_motor_command(motor_command_text):
    if motor_command_text == "q":
        print("user quit")
        return None
    try:
        motor_command = float(motor_command_text)
        motor_command = clamp(-1.0, 1.0, motor_command)
        return motor_command
    except ValueError:
        print("input command is not numerical, please enter a numerical motor command :")
        return "googoogaga"


while True:
    motor_command_text = input(
        "please enter a numerical motor command, or type q to quit: ")
    filtered_motor_command = parse_motor_command(motor_command_text)

    if filtered_motor_command is None:
        print(filtered_motor_command)
        break
    if filtered_motor_command == "googoogaga":
        continue

    print("valid input, filtered motor command is:", filtered_motor_command)

    break
