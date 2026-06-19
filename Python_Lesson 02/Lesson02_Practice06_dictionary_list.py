leg_config = {
    "front_right": {
        "hip": {"can_id": 3, "state": "disabled"},
        "thigh": {"can_id": 2, "state": "disabled"},
        "shank": {"can_id": 1, "state": "enabled"},
    },
    "front_left": {
        "hip": {"can_id": 6, "state": "disabled"},
        "thigh": {"can_id": 5, "state": "enabled"},
        "shank": {"can_id": 4, "state": "disabled"},
    },
    "rear_right": {
        "hip": {"can_id": 9,  "state": "enabled"},
        "thigh": {"can_id": 8, "state": "enabled"},
        "shank": {"can_id": 7, "state": "disabled"},
    },
    "rear_left": {
        "hip": {"can_id": 12, "state": "enabled"},
        "thigh": {"can_id": 11, "state": "disabled"},
        "shank": {"can_id": 10, "state": "disabled"},
    },
}

for leg_name, leg_configs in leg_config.items():
    for joint_name, joint_config in leg_configs.items():
        can_bus_id = joint_config["can_id"]
        joint_state = joint_config["state"]

        if joint_state == "enabled":
            print("joint name is:", leg_name, joint_name, can_bus_id)
