motor_id = 1
motor_name = "shank_1"


class Actuator_config:
    def __init__(self, motor_id, motor_name):
        self.motor_id = motor_id
        self.motor_name = motor_name


shank_motor = Actuator_config(motor_id, motor_name)
print(shank_motor.motor_id)
print(shank_motor.motor_name)
