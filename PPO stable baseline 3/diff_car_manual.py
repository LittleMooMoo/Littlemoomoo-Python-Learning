# import path library to help construct file paths
from pathlib import Path
# mujoco reads XML and create original model
import mujoco
import mujoco.viewer
# np for numerical calculations
import numpy as np
import time
import msvcrt


def clamp(low, max, value):
    if value <= low:
        value = low
    if value >= max:
        value = max
    return value


# __file__ is the path of this script
# .resolve converts it to an absolute path
# .parent gets the folder containing the script
# xml filename
xml_path = Path(__file__).resolve().parent / "4wheelcar.xml"
cpu_model = mujoco.MjModel.from_xml_path(str(xml_path))
cpu_data = mujoco.MjData(cpu_model)

chassis_id = cpu_model.body("chasis").id

# Output buffer: 3 angular components, then 3 linear components.
chassis_velocity = np.zeros(6, dtype=np.float64)

last_print = time.perf_counter()
# Controls have shape (number_of_worlds, number_of_actuators).
# cpu_model.nu returns the number of actuators
controls = np.zeros((1, cpu_model.nu), dtype=np.float32)

Fleft_qpos = int(cpu_model.joint("Fleft_axle").qposadr[0])
Fright_qpos = int(cpu_model.joint("Fright_axle").qposadr[0])

Fleft_motor = cpu_model.actuator("Fleft_motor_pos").id
Fright_motor = cpu_model.actuator("Fright_motor_pos").id

Rleft_qpos = int(cpu_model.joint("Rleft_axle").qposadr[0])
Rright_qpos = int(cpu_model.joint("Rright_axle").qposadr[0])

Rleft_motor = cpu_model.actuator("Rleft_motor_pos").id
Rright_motor = cpu_model.actuator("Rright_motor_pos").id

Fleft_target = float(cpu_data.qpos[Fleft_qpos])
Fright_target = float(cpu_data.qpos[Fright_qpos])

Rleft_target = float(cpu_data.qpos[Rleft_qpos])
Rright_target = float(cpu_data.qpos[Rright_qpos])

forward_command = 0.0
turn_command = 0.0
wheel_speed = 20.0  # Target-angle change in rad/s.

with mujoco.viewer.launch_passive(cpu_model, cpu_data) as viewer:
    while viewer.is_running():
        start = time.perf_counter()

        if msvcrt.kbhit():
            key = msvcrt.getwch().lower()
            # Keyboard handling:
            if key == "w":
                forward_command = 1.0
                turn_command = 0.0
            elif key == "s":
                forward_command = -1.0
                turn_command = 0.0
            elif key == "a":
                forward_command = 0.0
                turn_command = 1.0
            elif key == "d":
                forward_command = 0.0
                turn_command = -1.0
            elif key == " ":
                forward_command = 0.0
                turn_command = 0.0

        with viewer.lock():
            Langle_increment = (
                forward_command + turn_command
            ) * wheel_speed * cpu_model.opt.timestep

            Rangle_increment = (
                forward_command - turn_command
            ) * wheel_speed * cpu_model.opt.timestep

            Fleft_target += Langle_increment
            Fleft_target = clamp(-50, 50, Fleft_target)
            Fright_target += Rangle_increment
            Fright_target = clamp(-50, 50, Fright_target)

            Rleft_target += Langle_increment
            Rleft_target = clamp(-50, 50, Rleft_target)
            Rright_target += Rangle_increment
            Rright_target = clamp(-50, 50, Rright_target)

            cpu_data.ctrl[Fleft_motor] = Fleft_target
            cpu_data.ctrl[Fright_motor] = Fright_target
            cpu_data.ctrl[Rleft_motor] = Rleft_target
            cpu_data.ctrl[Rright_motor] = Rright_target

            mujoco.mj_step(cpu_model, cpu_data)
            mujoco.mj_forward(cpu_model, cpu_data)

            mujoco.mj_objectVelocity(
                cpu_model,
                cpu_data,
                mujoco.mjtObj.mjOBJ_BODY,
                chassis_id,
                chassis_velocity,
                0,
            )
            angular_velocity = chassis_velocity[:3].copy()
            linear_velocity = chassis_velocity[3:].copy()
        # Keep your velocity and wheel-angle measurements here.
        simulation_time = float(cpu_data.time)
        # 3. Update the viewer.
        viewer.sync()

        # 4. Avoid running faster than real time.
        remaining = cpu_model.opt.timestep - (
            time.perf_counter() - start
        )
        if remaining > 0:
            time.sleep(remaining)
