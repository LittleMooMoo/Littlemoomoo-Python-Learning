# import path library to help construct file paths
from pathlib import Path
# mujoco reads XML and create original model
import mujoco
import mujoco.viewer
# mjw runs physic under mujoco
import mujoco_warp as mjw
# np for numerical calculations
import numpy as np
# warp manage GPU devices and computations
import warp as wp
import time

amplitude = 8.0
frequency = 1.0
# init() initialize the warp and discovers available devices
wp.init()

# __file__ is the path of this script
# .resolve converts it to an absolute path
# .parent gets the folder containing the script
# xml filename
xml_path = Path(__file__).resolve().parent / "diff_car.xml"
cpu_model = mujoco.MjModel.from_xml_path(str(xml_path))
cpu_data = mujoco.MjData(cpu_model)

with wp.ScopedDevice("cuda:0"):
    chassis_id = cpu_model.body("chasis").id

    # Output buffer: 3 angular components, then 3 linear components.
    chassis_velocity = np.zeros(6, dtype=np.float64)

    last_print = time.perf_counter()
    # Transfer the model and create one simulated world.
    gpu_model = mjw.put_model(cpu_model)
    # nworld = 1 means simulate one copy of the car
    gpu_data = mjw.make_data(cpu_model, nworld=1)
    # Controls have shape (number_of_worlds, number_of_actuators).
    # cpu_model.nu returns the number of actuators
    controls = np.zeros((1, cpu_model.nu), dtype=np.float32)

    next_print_time = 0.0
    print_interval = 0.1
    with mujoco.viewer.launch_passive(cpu_model, cpu_data) as viewer:
        while viewer.is_running():
            start = time.perf_counter()
            t = cpu_data.time
            pos_target = amplitude*np.sin(2 * np.pi * frequency*t)
            # Your existing PD setup: target angle 1 rad, target speed 0.
            # first 0 represent the car at world 0
            controls[0, cpu_model.actuator(
                "left_motor_pos").id] = pos_target
            controls[0, cpu_model.actuator(
                "right_motor_pos").id] = pos_target

            wp.copy(gpu_data.ctrl, wp.array(controls, dtype=wp.float32))

            # 1. Advance physics on the GPU.
            mjw.step(gpu_model, gpu_data)

            # 2. Copy world 0's state into the CPU viewer data.
            with viewer.lock():
                mjw.get_data_into(
                    cpu_data,
                    cpu_model,
                    gpu_data,
                    world_id=0,
                )
                # Refresh geometry positions for the copied state.
                # forward calculate quantity at the current state
                mujoco.mj_forward(cpu_model, cpu_data)
                mujoco.mj_objectVelocity(
                    cpu_model,
                    cpu_data,
                    mujoco.mjtObj.mjOBJ_BODY,  # Measure at this body's COM
                    chassis_id,
                    chassis_velocity,
                    0,                       # Express components along WORLD axes
                )

                angular_velocity = chassis_velocity[:3].copy()  # rad/s
                linear_velocity = chassis_velocity[3:].copy()   # m/s

                speed = np.linalg.norm(linear_velocity)          # m/s
                angular_speed = np.linalg.norm(angular_velocity)  # rad/s

            if t > next_print_time:
                print(f"angular velocity:{angular_velocity}\n")
                print(f"linear velocity:{linear_velocity}\n")
            next_print_time += print_interval
            # 3. Update the viewer.
            viewer.sync()

            # 4. Avoid running faster than real time.
            remaining = cpu_model.opt.timestep - (
                time.perf_counter() - start
            )
            if remaining > 0:
                time.sleep(remaining)
