import mujoco
import numpy as np
import torch
from pathlib import Path
import warp as wp
import mujoco_warp as mjw

wp.init()
num_of_envs = 16
xml_path = Path(__file__).resolve().parent/"diff_car.xml"
car = mujoco.MjModel.from_xml_path(str(xml_path))
# car_state can contain multiple data
car_state = mujoco.MjData(car)

with wp.ScopedDevice("cuda:0"):
    car1 = mjw.put_model(car)
    car_state1 = mjw.make_data(car, nworld=num_of_envs)
    # Alternative, when data contains a desired initial state:
    # gpu_data = mjw.put_data(model, data, nworld=128)
    car_state1.ctrl.fill_(200.0)
    for step in range(10):
        mjw.step(car1, car_state1)

    left_joint = car.joint("left_axle")
    right_joint = car.joint("right_axle")
    # use this line to get the index where the left joint qpos is stored in the np array of the array
    left_qpos_index = int(left_joint.qposadr[0])
    # convert into pytorch tensor
    car_qpos_tensor = wp.to_torch(car_state1.qpos)

    # qpos is defined as [number of env, number of position values]
    # 4. Select that wheel across all environments.
    # qpos itself is the joint position of all joints in an environment,
    # left_joint_qpos_index can extract the part only for left joint,
    #  then put all 16 environment's same joint's qpos into one tensor for better large batch training;
    left_angles = car_qpos_tensor[:, left_qpos_index:left_qpos_index + 1]
    print("initial left axle angles: ", left_angles)
    # these two functions can search and return the index for the position
    # and velocity vecotr begining index in the stored state array or this specific joint
    # q_index = int(left_joint.qposadr[0])
    # v_index = int(left_joint.dofadr[0])

    left_id = car.actuator("left_motor_pos").id
    right_id = car.actuator("right_motor_pos").id
    body_id = car.body("chasis").id  # spelling in your original XML

    # value, not the address q_index

    # car_state.actuator("left_motor_pos").ctrl[:] = 10.0
    # mujoco.mj_step(car, car_state)
    # mujoco.mj_forward(car, car_state)

    # # these are in radians and radians/sec
    # qL = car_state.joint("left_axle").qpos
    # wL = car_state.joint("left_axle").qvel
    # # body frame origin position in world frame
    # p_chasis = car_state.body("chasis").xpos
    # # body COM position in world frame
    # p_chasis_COM = car_state.body("chasis").xipos
    # p_quat = car_state.body("chasis").xquat

    # Wait for the GPU to finish before reporting results.
    # Wait for pending GPU calculations to finish.
    wp.synchronize_device()

    print("Position array shape:", car_state1.qpos.shape)
    print("Control array shape:", car_state1.ctrl.shape)

    # Copy the simulation times from GPU to CPU for printing.
    print("Simulation times:", car_state1.time.numpy())

    print("Left axle angles:", left_angles)
    print("Left axle angles shape:", tuple(left_angles.shape))

# left_angle_tensor = torch.from_numpy(
#     car_state.sensor("left_wheel_angle").data.copy())
# right_angle_tensor = torch.from_numpy(
#     car_state.sensor("right_wheel_angle").data.copy())

# left_speed_tensor = torch.from_numpy(
#     car_state.sensor("left_wheel_speed").data.copy())
# right_speed_tensor = torch.from_numpy(
#     car_state.sensor("right_wheel_speed").data.copy())

# imu_gyro_tensor = torch.from_numpy(
#     car_state.sensor("imu_gyro").data.copy())
# imu_lin_accel_tensor = torch.from_numpy(
#     car_state.sensor("imu_lin_accel").data.copy())
# imu_quat_tensor = torch.from_numpy(
#     car_state.sensor("imu_quat").data.copy())

# measurement_tensor = torch.cat([
#     imu_quat_tensor,
#     imu_gyro_tensor,
#     imu_lin_accel_tensor,
#     left_angle_tensor,
#     right_angle_tensor], dim=0)
# measurement_tensor = measurement_tensor.to('cuda')
