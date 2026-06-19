from dataclasses import dataclass


@dataclass(frozen=True)
class position:
    name: str
    foot_position: tuple


neutral_angle = position(name="neutral_angle",
                         foot_position=(-20.47, 0, 358.0))
homing_angle = position(name="homing_angle",
                        foot_position=(0, 0, 380))

foot_position_text = input("choose a foot pose to output, or type q to quit:")


def get_footpose(foot_position_text):
    foot_position_text = foot_position_text
    if foot_position_text == "q":
        print("user quit")
        foot_pose = None
    elif foot_position_text == "neutral_angle":
        x1, y1, z1 = neutral_angle.foot_position
        foot_pose = [x1, y1, z1]
        print("selected foot position is:", foot_pose)
    elif foot_position_text == "homing_angle":
        x2, y2, z2 = homing_angle.foot_position
        foot_pose = [x2, y2, z2]
        print("selected foot pose is:", foot_pose)

    else:
        print("no found")
        return None
    return foot_pose


get_footpose(foot_position_text)
