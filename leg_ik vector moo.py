from quadruped_leg_ik import leg_ik
import numpy as np


def leg_cartesian_coordinate(x, y, z):
    return np.array([x, y, z])


def leg_polar_coordinate(theta_h_rad, theta_t_rad, theta_s_rad):

    theta_h_rad, _, _ = leg_ik(x, y, z)
    _, theta_t_rad, _ = leg_ik(x, y, z)
    _, _, theta_s_rad = leg_ik(x, y, z)

    return np.array([theta_h_rad, theta_t_rad, theta_s_rad])


user_input = input("Enter the desired leg cartesian coordinates (x y z):")

input_coordinates = user_input.split()

input_x = float(input_coordinates[0])
input_y = float(input_coordinates[1])
input_z = float(input_coordinates[2])

input_cartesian_coordinates = np.array([input_x, input_y, input_z])

print("Input Cartesian Coordinates:", input_cartesian_coordinates)

print("Calculating Polar Coordinates...")

x = input_x
y = input_y
z = input_z

Calculated_polar_coordinates = leg_polar_coordinate(x, y, z)
print("polar coordinates:", Calculated_polar_coordinates)

polar_coordiantes_deg = np.degrees(Calculated_polar_coordinates)
print("Polar Coordinates (degrees):", polar_coordiantes_deg)
