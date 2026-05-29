class ActuatorConfig:
    def __init__(self, motor_id, motor_name):
        self.motor_id = motor_id
        self.motor_name = motor_name


hip_front_left = ActuatorConfig(3, "hip_front_left")
thigh_front_left = ActuatorConfig(2, "thigh_front_left")
shank_front_left = ActuatorConfig(1, "shank_front_left")

user_input = input("please type in can id or motor name :")
search_by_id = user_input.isdigit()

if search_by_id:
    searched_can_id = int(user_input)
    print("search by can id")
else:
    searched_motor_name = user_input
    print("search by motor name")

actuators = [hip_front_left, thigh_front_left, shank_front_left]
found_motor = None

for actuator in actuators:
    if search_by_id:
        if actuator.motor_id == searched_can_id:
            found_motor = actuator
            break
    else:
        if actuator.motor_name == searched_motor_name:
            found_motor = actuator
            break


if found_motor is None:
    print("no motor found")
else:
    print(found_motor.motor_id)
    print(found_motor.motor_name)
