"""
python_lesson_03_04_robot_ready.py

Lesson 03 + Lesson 04 combined.

Lines starting with # are comments. Python ignores them.
You can run this file directly.
"""

from dataclasses import dataclass


# ============================================================
# LESSON 03: Better Robot Data And Safer Functions
# ============================================================

# A tuple is good for fixed data.
# Example: neutral foot position: x, y, z
neutral_position = (0.0, 84.26, 353.3)


def print_position(position: tuple[float, float, float]) -> None:
    x, y, z = position
    print("x =", x)
    print("y =", y)
    print("z =", z)


def clamp(value: float, min_value: float, max_value: float) -> float:
    if value < min_value:
        return min_value
    if value > max_value:
        return max_value
    return value


def parse_float(text: str) -> float:
    try:
        return float(text)
    except ValueError:
        raise ValueError("Input must be a number")


def parse_can_id(text: str) -> int:
    try:
        return int(text)
    except ValueError:
        raise ValueError("CAN ID must be a number")


@dataclass
class JointConfig:
    can_id: int
    state: str
    kp: float
    kd: float
    torque_limit: float


def is_enabled(joint: JointConfig) -> bool:
    return joint.state == "enabled"


hip_config = JointConfig(
    can_id=3,
    state="enabled",
    kp=0.25,
    kd=0.015,
    torque_limit=1.30,
)

thigh_config = JointConfig(
    can_id=2,
    state="enabled",
    kp=0.25,
    kd=0.015,
    torque_limit=1.30,
)

shank_config = JointConfig(
    can_id=1,
    state="disabled",
    kp=0.25,
    kd=0.015,
    torque_limit=1.30,
)


# ============================================================
# LESSON 04: Classes, Methods, And Fake Motor Controller
# ============================================================

class Motor:
    def __init__(self, name: str, can_id: int):
        self.name = name
        self.can_id = can_id
        self.enabled = False
        self.position = 0.0

    def enable(self) -> None:
        self.enabled = True
        print(self.name, "enabled")

    def disable(self) -> None:
        self.enabled = False
        print(self.name, "disabled")

    def move_to(self, target_position: float) -> None:
        if not self.enabled:
            print(self.name, "is disabled. Cannot move.")
            return

        target_position = clamp(target_position, -10.0, 10.0)
        self.position = target_position
        print(self.name, "moved to", self.position)


class Leg:
    def __init__(self, name: str, hip: Motor, thigh: Motor, shank: Motor):
        self.name = name
        self.hip = hip
        self.thigh = thigh
        self.shank = shank

    def enable_all(self) -> None:
        self.hip.enable()
        self.thigh.enable()
        self.shank.enable()

    def disable_all(self) -> None:
        self.hip.disable()
        self.thigh.disable()
        self.shank.disable()

    def get_all_motors(self) -> list[Motor]:
        return [self.hip, self.thigh, self.shank]


front_right_hip = Motor("front_right_hip", 3)
front_right_thigh = Motor("front_right_thigh", 2)
front_right_shank = Motor("front_right_shank", 1)

front_right_leg = Leg(
    "front_right",
    front_right_hip,
    front_right_thigh,
    front_right_shank,
)

motors = front_right_leg.get_all_motors()


# ============================================================
# RUNNABLE TEST CODE
# ============================================================

print("Neutral position:")
print_position(neutral_position)

print()
print("Joint config test:")
print("hip enabled?", is_enabled(hip_config))
print("shank enabled?", is_enabled(shank_config))

print()
print("Enabling all motors...")
front_right_leg.enable_all()

while True:
    can_id_text = input("Enter CAN ID, or q to quit: ")

    if can_id_text == "q":
        print("user quit")
        break

    try:
        target_can_id = parse_can_id(can_id_text)
    except ValueError as error:
        print(error)
        continue

    position_text = input("Enter target position: ")

    try:
        target_position = parse_float(position_text)
    except ValueError as error:
        print(error)
        continue

    found_motor = None

    for motor in motors:
        if motor.can_id == target_can_id:
            found_motor = motor
            break

    if found_motor is None:
        print("No motor with that CAN ID")
        continue

    found_motor.move_to(target_position)


# ============================================================
# PRACTICE PROBLEMS
# ============================================================

# Practice 1:
# Create another tuple called sleepy_position with x, y, z.

# Practice 2:
# Create another JointConfig for rear_right_hip.

# Practice 3:
# Create three more Motor objects for front_left leg.

# Practice 4:
# Create a front_left_leg object using the Leg class.

# Practice 5:
# Put all motors from front_right and front_left into one big list.

# Practice 6:
# Ask user for CAN ID and print the motor name if found.

# Practice 7:
# Add a method to Motor called move_relative(self, delta).

# Practice 8:
# Add fake trajectory movement:
# for step in range(10):
#     front_right_hip.move_to(step * 0.1)