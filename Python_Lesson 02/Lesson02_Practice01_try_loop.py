# Lesson 02 Practice Exercise 01

input_angle = input("Enter an angle in degrees: ")


def parse_input_angle(input_angle):

    try:
        angle_deg = float(input_angle)
        return angle_deg

    except ValueError:
        print("Invalid input, please enter an angle in degrees ")
        return None


angle_deg = parse_input_angle(input_angle)

if angle_deg is not None:
    if -45 <= angle_deg <= 45:
        print("input angle is safe, input angle is :", angle_deg)
    else:
        print("input angle is unsafe, please enter another angle")
