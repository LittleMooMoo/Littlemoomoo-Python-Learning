class MotorController():
    def __init__(self, id, torque_limit):
        self.id = id
        self.torque_limit = torque_limit


class JointController(MotorController):
    def __init__(self, motor_id, torque_limit, gear_ratio):
        super().__init__(motor_id, torque_limit)
        self.gear_ratio = gear_ratio


@dataclass
{

}

joint1 = JointController(1, 2, 25)
print(joint1.id)            # 1
print(joint1.torque_limit)  # 2
