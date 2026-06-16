import math
import numpy as np

a = np.array([[1, 2, 5, 15],
              [5, 9, 10, 3],
              [2, 3, 4, 8]])
b = np.array([[3, 5, 7, 9],
              [1, 3, 6, 4],
              [7, 7, 9, 0]])
print(a@b.T)

def joint_vector(theta_h, theta_t, theta_s):
    theta_h = math.radians(theta_h)
    theta_t = math.radians(theta_t)
    theta_s = math.radians(theta_s)
    
    return np.array([theta_h, theta_t, theta_s])
    
print(joint_vector(30, 45, 60))