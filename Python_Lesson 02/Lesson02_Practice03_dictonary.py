motor_can_id = {
    "front_right_hip": 3,
    "front_right_thigh": 2,
    "front_right_shank": 1,
}


def parse_can_id(target_id_text):
    try:
        numerical_target_id = int(target_id_text)
        return numerical_target_id
    except ValueError:
        return None


while True:
    target_id_text = input("please enter the CAN id desired:")
    target_id = parse_can_id(target_id_text)

    if target_id is None:
        print("please enter a numerical can id,")
        continue

    print("searching....")

    found = False

    for leg_name, joint_id in motor_can_id.items():
        if target_id == joint_id:
            print("motor found,motor id is:", leg_name)
            found = True
            break

    if not found:
        print("no motor found")

    break
