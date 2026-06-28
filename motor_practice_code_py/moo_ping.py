import berkeley_humanoid_lite_lowlevel as recoil

can_id_mapping = {
    "front_right_leg": {
        "hip": 3, "thigh": 2, "shank": 1
    },
    "front_left_leg": {
        "hip": 6, "thigh": 5, "shank": 4
    },
    "rear_right_leg": {
        "hip": 9, "thigh": 8, "shank": 7
    },
    "rear_left_leg": {
        "hip": 12, "thigh": 11, "shank": 10
    }
}

timeout_threshold = 0.1


def get_motor_id():
    motor_id_list = []
    for leg_name, leg_data in can_id_mapping.items():
        for joint_name, motor_id in leg_data.items():
            motor_id_list.append(motor_id)

    return motor_id_list


def ping_robot(bus):
    all_motors_online = True
    motor_id_list = get_motor_id()
    for motor_id in motor_id_list:
        if (bus.ping(motor_id, timeout=timeout_threshold)):
            print({motor_id}, "is online")
        else:
            print({motor_id}, "is offline")


def main() -> None:
    motor_id_list = get_motor_id()
    print(motor_id_list)


if __name__ == "__main__":
    main()
