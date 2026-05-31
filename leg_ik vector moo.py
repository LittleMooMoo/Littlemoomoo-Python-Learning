# This code is for moo moo cow to practice math
from quadruped_leg_ik import clamp, leg_ik
import numpy as np


def leg_polar_coordinate(x, y, z):

    theta_h_rad_raw, theta_t_rad_raw, theta_s_rad_raw = leg_ik(x, y, z)

    theta_h_rad = clamp(theta_h_rad_raw, -np.pi, np.pi)
    theta_t_rad = clamp(theta_t_rad_raw, -np.pi, np.pi)
    theta_s_rad = clamp(theta_s_rad_raw, -np.pi, np.pi)
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

calculated_polar_coordinates = leg_polar_coordinate(x, y, z)
print("polar coordinates:", calculated_polar_coordinates)

polar_coordiantes_deg = np.degrees(calculated_polar_coordinates)
print("Polar Coordinates (degrees):", polar_coordiantes_deg)

target_polar_coordinates_deg = np.array([0, 45, 90])


def position_error(target_polar_coordinates_deg, polar_coordinates_deg):

    error_raw = target_polar_coordinates_deg - polar_coordinates_deg

    filtered_error = abs(error_raw)

    return error_raw, filtered_error


error_raw, filtered_error = position_error(
    target_polar_coordinates_deg, polar_coordiantes_deg)

if np.any(np.abs(error_raw) > 25):
    print("Error exceeds threshold, unreachable")
else:
    print("position is reachable, error within threshold: ", filtered_error)
