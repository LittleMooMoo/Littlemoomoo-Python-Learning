# MuJoCo and Isaac Lab: custom environments, RL training, evaluation, and deployment

Prepared for Jiaxi on **2026-10-20**. Examples use Python unless labelled XML, PowerShell, or pseudocode.

This is a practical reference to the APIs needed for the complete workflow, rather than an exhaustive listing of every function in both projects. It covers ordinary rigid-body robots, a small wheeled car, and the additional state/control conventions needed for a quadruped. Cameras and contacts have introductory entries; advanced rendering, deformables, and distributed training are outside this guide.

**How to read this guide:** start with sections 1–3, run the complete MuJoCo example in section 6, then study the Isaac Lab sections. Short snippets labelled *fragment* need the objects described in their surrounding text. Only blocks labelled *complete file* are intended to run independently. Names such as `wheel`, `robot.usd`, and `my_tasks` in fragments are examples to replace, not files or objects this document creates.

## Contents

1. [Software layers and versions](#1-software-layers-and-versions)
2. [Time, shapes, units, and coordinates](#2-time-shapes-units-and-coordinates)
3. [MuJoCo CPU API](#3-mujoco-cpu-api)
4. [MJCF model and actuator API](#4-mjcf-model-and-actuator-api)
5. [MuJoCo Warp API](#5-mujoco-warp-api)
6. [Gymnasium API and a complete custom environment](#6-gymnasium-api-and-a-complete-custom-environment)
7. [Stable-Baselines3 PPO training and evaluation](#7-stable-baselines3-ppo-training-and-evaluation)
8. [Isaac Lab setup, assets, and simulation](#8-isaac-lab-setup-assets-and-simulation)
9. [Isaac Lab state, control, and reset APIs](#9-isaac-lab-state-control-and-reset-apis)
10. [Isaac Lab custom environment hooks](#10-isaac-lab-custom-environment-hooks)
11. [Manager-based environments and sensors](#11-manager-based-environments-and-sensors)
12. [Isaac Lab training, playback, and export](#12-isaac-lab-training-playback-and-export)
13. [Deployment and matching the training contract](#13-deployment-and-matching-the-training-contract)
14. [Debugging reference](#14-debugging-reference)
15. [Validation and source references](#15-validation-and-source-references)

## 1. Software layers and versions

The two main routes in this guide are:

```text
MuJoCo CPU -> your Gymnasium Env -> Stable-Baselines3 PPO

Isaac Lab DirectRLEnv/ManagerBasedRLEnv -> Newton backend with MJWarp
                                     -> RSL-RL PPO
```

MuJoCo and MuJoCo Warp simulate physics. Gymnasium defines an environment interface. Isaac Lab supplies environment orchestration, batched assets, sensors, resets, and training integration. PPO is an optimization algorithm implemented by a training library. A simulator does not automatically invent the reward or train a policy.

Standalone `mujoco_warp` is also usable without Isaac Lab. Its arrays and stepping functions are **not** interchangeable with Isaac Lab articulation objects.

### 1.1 Verified local version snapshot

These values were read from the installed environments while preparing this file. They are a snapshot, not a recommendation to upgrade packages independently.

| Environment | Package | Installed version |
|---|---|---|
| Learning repository `.venv` | `mujoco` | 3.14.0 |
| Same | `mujoco-warp` | 3.16.0 |
| Same | `warp-lang` | 1.18.0 |
| Same | `gymnasium` | 1.3.0 |
| Same | `stable-baselines3` | 2.9.0 |
| Same | `torch` | 2.13.0+cu132 |
| `C:\Users\jiaxi\IsaacLab\.venv` | `isaaclab` distribution metadata | 40.0.1 |
| Same | repository `VERSION` file | 3.0.0 |
| Same | `newton` | 1.8.0.dev0 |
| Same | `warp-lang` | 1.17.0 |
| Same | `rsl-rl-lib` | 5.5.1 |
| Same | `torch` | 2.11.0+cu130 |

Isaac Lab source revision read from its local branch ref: `5ffd3d5ac0b4d77d31ea820930a8f5fcac673d87`. The distribution version and repository release version differ; source inspection, not the package number alone, determined the examples below. Uncommitted source modifications were not audited.

**Version boundary:** this checkout uses `.torch` views, `xyzw` task quaternions, the actuator command interface, and the unified `isaaclab train/play` CLI. Older 1.x/2.x tutorials commonly use direct tensors, `wxyz`, `AppLauncher`, and separate training scripts. Do not mix those recipes without adapting them. See the [official migration guide](https://isaac-sim.github.io/IsaacLab/develop/source/isaaclab/migrating.html).

### 1.2 Interpreter selection and package inspection

`importlib.metadata.version(name)` reads distribution metadata. `sys.executable` identifies the interpreter actually running the file.

```python
import sys
from importlib.metadata import version
print(sys.executable)
print(version("mujoco"))
```

Use the intended executable explicitly in PowerShell; activation is optional when doing this:

```powershell
& 'C:\Users\jiaxi\OneDrive\Documentos\GitHub\Littlemoomoo-Python-Learning\.venv\Scripts\python.exe' your_script.py
& 'C:\Users\jiaxi\IsaacLab\.venv\Scripts\python.exe' your_isaac_script.py
```

Keep these environments separate. Do not copy a `.venv` or an entire simulator installation into the other environment.

## 2. Time, shapes, units, and coordinates

### 2.1 Three different clocks

Let `physics_dt` be one integration step and `decimation` be the number of physics steps per policy action. Then:

```python
physics_dt = 0.002          # seconds; 500 Hz physics
decimation = 10
control_dt = physics_dt * decimation  # 0.020 seconds; 50 Hz policy
```

Training can run faster or slower than real time. Rewards, episode durations, and controllers should use simulation time. Wall-clock sleep is useful for watching or real hardware, not headless training. Viewer refresh can be much slower than physics.

### 2.2 Shape conventions

| Quantity | One MuJoCo/Gymnasium world | Batched Isaac Lab or MJWarp |
|---|---|---|
| Policy action | `(A,)` | `(N, A)` |
| Vector observation | `(O,)` | `(N, O)` |
| Joint positions | selected elements of `(nq,)` | usually `(N, J)` for articulation joints |
| Reward | Python `float` | tensor `(N,)` |
| Termination | Python `bool` | boolean tensor `(N,)` |

`N` = environments, `A` = policy outputs, `O` = observations, `J` = selected joints. Four wheels can use two actions if each action drives one side. Twelve quadruped joints can use twelve position-target actions even though the motor controller also uses gains, velocity targets, and feedforward torque.

### 2.3 Coordinate conventions that must remain explicit

| Item | Convention in this guide |
|---|---|
| Length, mass, time | meters, kilograms, seconds |
| Hinge position/velocity | radians, radians/second |
| Torque | N·m at the relevant joint |
| MuJoCo quaternion | `(w, x, y, z)` |
| This Isaac Lab checkout's task quaternion | `(x, y, z, w)` |
| `mj_objectVelocity` output | angular XYZ, then linear XYZ |
| Isaac Lab root velocity arrays | linear XYZ, then angular XYZ |
| `_w` / `_b` in Isaac Lab | world coordinates / body coordinates |

Example conversion, with a batch of MuJoCo quaternions `q_mj`:

```python
q_lab = q_mj[..., [1, 2, 3, 0]]   # wxyz -> xyzw
q_mj_again = q_lab[..., [3, 0, 1, 2]]
```

The reference point matters too: velocity at a body's origin can differ from velocity at its COM when it rotates. Specify both the point and the axes used to express the vector.

## 3. MuJoCo CPU API

The entries below use the native `mujoco` package, not `mujoco-py`. Reference: [Python bindings](https://mujoco.readthedocs.io/en/stable/python.html) and the installed package's function docstrings.

### 3.1 `MjModel.from_xml_path`, `MjModel.from_xml_string`, and `MjData`

`MjModel` holds the compiled robot and simulation parameters. `MjData` holds one world's changing state, controls, and computed results. Create multiple `MjData` objects to maintain independent states for the same model. Loading a model does not advance time.

```python
from pathlib import Path
import mujoco
import numpy as np

xml_path = Path(__file__).resolve().parent / "diff_car.xml"
model = mujoco.MjModel.from_xml_path(str(xml_path))
data = mujoco.MjData(model)

# Alternative: compile XML text already in memory.
other_model = mujoco.MjModel.from_xml_string(
    '<mujoco><worldbody><geom type="sphere" size="0.1"/></worldbody></mujoco>'
)
other_data = mujoco.MjData(other_model)
```

Python variables called `cpu_model` and `cpu_data` are simply names you chose for these objects. There is no `model.data` member that replaces a separate `MjData`.

### 3.2 Model dimensions and options

Each row gives the meaning and a short usage example. These fields belong to `model`.

| Member | Meaning | Example |
|---|---|---|
| `nq` | Number of position coordinates, including quaternion components | `print(model.nq)` |
| `nv` | Number of velocity degrees of freedom | `dq = np.zeros(model.nv)` |
| `nu` | Number of control inputs | `u = np.zeros(model.nu)` |
| `njnt` | Number of joint objects, not scalar DOFs | `print(model.njnt)` |
| `nbody` | Number of bodies, including world body | `print(model.nbody)` |
| `ngeom` | Number of collision/visual geometries | `print(model.ngeom)` |
| `nsensordata` | Total scalar sensor output count | `print(model.nsensordata)` |
| `opt.timestep` | Physics integration time step in seconds | `dt = float(model.opt.timestep)` |
| `opt.gravity` | World gravity XYZ | `model.opt.gravity[:] = [0, 0, -9.81]` |

A free joint contributes **7** position coordinates (XYZ + quaternion) and **6** velocity coordinates. Consequently `nq != nv` is normal. A joint ID is neither a position-array address nor a velocity-array address.

### 3.3 Named access, IDs, and addresses

`model.joint(name)`, `model.body(name)`, and `model.actuator(name)` return named views. Their `.id` is an object ID. Joint `.qposadr` and `.dofadr` locate the start of their state in `qpos` and `qvel`.

```python
# Fragment: these names must exist in your XML.
j = model.joint("left_axle")
q_index = int(j.qposadr[0])
v_index = int(j.dofadr[0])
motor_id = model.actuator("left_motor_pos").id
body_id = model.body("chasis").id  # spelling in your original XML

angle = float(data.qpos[q_index])  # value, not the address q_index
omega = float(data.qvel[v_index])
```

`mj_name2id` is the lower-level name lookup and returns `-1` when absent. `mj_id2name` does the reverse and can return `None` for unnamed objects.

```python
body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "chasis")
if body_id < 0:
    raise KeyError("chasis is not in this model")
print(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id))
```

### 3.4 State and actuation members on `MjData`

| Member | Meaning | Example |
|---|---|---|
| `time` | Simulation time in seconds | `t = float(data.time)` |
| `qpos` | Generalized position, `(nq,)` | `q_saved = data.qpos.copy()` |
| `qvel` | Generalized velocity, `(nv,)` | `dq_saved = data.qvel.copy()` |
| `qacc` | Computed generalized acceleration | `print(data.qacc.copy())` |
| `ctrl` | Actuator command input, `(nu,)`; units depend on actuator | `data.ctrl[motor_id] = 0.2` |
| `act` | Internal activation states, if the actuator has dynamics | `print(data.act.shape)` |
| `actuator_force` | Computed scalar force in actuator coordinates | `print(data.actuator_force[motor_id])` |
| `qfrc_actuator` | Actuator contribution in generalized force coordinates | `print(data.qfrc_actuator[v_index])` |
| `qfrc_applied` | User-supplied generalized force | `data.qfrc_applied[v_index] = 0.1` |
| `xfrc_applied` | External body force XYZ then torque XYZ, world-oriented | `data.xfrc_applied[body_id] = [1, 0, 0, 0, 0, 0]` |

Array access is usually a **view into live simulator memory**. Use `.copy()` to retain a measurement over time. Writing an unrelated NumPy array named `controls` does not command a CPU simulation: write `data.ctrl[:] = controls`.

### 3.5 `mj_step` versus `mj_forward`

`mj_step(model, data)` evaluates dynamics and integrates forward in time. `mj_forward(model, data)` recomputes derived quantities at the current state without advancing time. Use forward after directly changing positions/velocities, and when you need derived measurements aligned with the latest integrated state.

```python
data.ctrl[:] = 0
mujoco.mj_step(model, data)           # advances one physics timestep
mujoco.mj_forward(model, data)        # refreshes current-state quantities

# Hold the same controls for 10 physics steps: one policy action.
mujoco.mj_step(model, data, nstep=10)
mujoco.mj_forward(model, data)
```

Calling `mj_forward` every substep when no refreshed measurement is needed adds work. Neither function trains a policy. There is no native `mujoco.step()` equivalent to these named functions.

### 3.6 `mj_resetData`, `mj_resetDataKeyframe`, and state snapshots

`mj_resetData` restores initial simulation data. It does not implement task randomization. `mj_resetDataKeyframe` resets from an XML keyframe by index.

```python
mujoco.mj_resetData(model, data)
data.qpos[q_index] += 0.05
mujoco.mj_forward(model, data)

# Only when the XML contains a keyframe:
if model.nkey > 0:
    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)
```

For a more complete simulation checkpoint, `mj_stateSize` determines buffer length, `mj_getState` fills the buffer, and `mj_setState` restores it. The state specification selects which fields are included. This is separate from saving a learned policy.

```python
spec = mujoco.mjtState.mjSTATE_INTEGRATION
snapshot = np.empty(mujoco.mj_stateSize(model, spec))
mujoco.mj_getState(model, data, snapshot, spec)
mujoco.mj_step(model, data)
mujoco.mj_setState(model, data, snapshot, spec)
mujoco.mj_forward(model, data)
```

### 3.7 Body, COM, site, and orientation measurements

These arrays contain world-frame results after forward kinematics.

| Member | Meaning | Example |
|---|---|---|
| `data.xpos` | Body-frame origin positions | `p = data.xpos[body_id].copy()` |
| `data.xipos` | Body inertial COM positions | `com = data.xipos[body_id].copy()` |
| `data.xquat` | Body orientation, wxyz | `q = data.xquat[body_id].copy()` |
| `data.xmat` | Flattened 3×3 body rotation matrix | `R = data.xmat[body_id].reshape(3, 3).copy()` |
| `data.site_xpos` | Site positions | `p = data.site_xpos[model.site("tip").id].copy()` |
| `data.subtree_com` | Mass-weighted COM of a body and descendants | `p = data.subtree_com[body_id].copy()` |

`xipos[body_id]` measures that body's COM; `subtree_com[body_id]` can include the attached wheels or legs. They need not be equal.

### 3.8 `mj_objectVelocity`: angular and linear velocity

This function writes into an output buffer and returns `None`. With `mjOBJ_BODY`, it measures the body's inertial frame/COM. `flg_local=0` expresses the vector in world axes; `1` uses local axes.

```python
velocity6 = np.zeros(6, dtype=np.float64)
mujoco.mj_forward(model, data)
mujoco.mj_objectVelocity(
    model, data, mujoco.mjtObj.mjOBJ_BODY, body_id, velocity6, 0
)
angular_velocity_w = velocity6[:3].copy()    # rad/s
linear_velocity_w = velocity6[3:].copy()     # m/s
speed = np.linalg.norm(linear_velocity_w)   # sqrt(vx² + vy² + vz²)

# Convert a world vector into body-link axes using the body's rotation.
R = data.xmat[body_id].reshape(3, 3)
linear_velocity_b = R.T @ linear_velocity_w
```

For velocity at a named marker, use `mjtObj.mjOBJ_SITE` and its site ID. Do not assume a raw slice of a free joint's `qvel` has every frame/reference-point convention your observation requires.

### 3.9 `mj_jacSite` and `mj_jacBodyCom`

A Jacobian maps generalized velocity to point linear or angular velocity. Both functions fill two `(3, nv)` arrays. Site Jacobians are useful for foot/hand control; a body COM Jacobian is useful for COM velocity and force mappings.

```python
jacp = np.zeros((3, model.nv))
jacr = np.zeros((3, model.nv))
mujoco.mj_jacBodyCom(model, data, jacp, jacr, body_id)
com_velocity_w = jacp @ data.qvel

# Fragment: requires an XML site named "tip".
site_id = model.site("tip").id
mujoco.mj_jacSite(model, data, jacp, jacr, site_id)
tip_velocity_w = jacp @ data.qvel
```

### 3.10 Contacts: `ncon`, `contact`, and `mj_contactForce`

`data.ncon` is the active contact count. Each `data.contact[i]` records its geometry IDs, position, and contact frame. `mj_contactForce` gives force followed by torque in that contact frame, not world axes.

```python
for i in range(data.ncon):
    contact = data.contact[i]
    wrench = np.zeros(6)
    mujoco.mj_contactForce(model, data, i, wrench)
    print(contact.geom1, contact.geom2, contact.pos.copy(), wrench[:3])
```

A contact between two bodies is not automatically task failure: wheel-ground and foot-ground contacts are expected. Filter by the specific geometries or body pairs you care about.

### 3.11 Sensors: `sensor_adr`, `sensor_dim`, and `sensordata`

Sensor outputs share one flat buffer. A sensor's address and dimension identify its slice. For a `<gyro name="imu_gyro" .../>` defined in XML:

```python
sid = model.sensor("imu_gyro").id
start = int(model.sensor_adr[sid])
count = int(model.sensor_dim[sid])
gyro = data.sensordata[start:start + count].copy()
```

Sensor frame and physical meaning come from the sensor type. An accelerometer's reading is not simply the world-coordinate second derivative of position; it measures proper acceleration in its local frame.

### 3.12 `viewer.launch`, `launch_passive`, `lock`, `sync`, and `close`

`launch()` starts a managed viewer and blocks your code. `launch_passive(model, data)` lets your code own the stepping loop. `is_running()` checks window status; `sync()` exchanges viewer state; `lock()` protects shared access; `close()` releases the viewer. The context manager closes it automatically.

```python
import time
import mujoco.viewer

with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        wall_start = time.perf_counter()
        with viewer.lock():
            data.ctrl[:] = 0
            mujoco.mj_step(model, data)
            mujoco.mj_forward(model, data)
            position = data.xpos[body_id].copy()
        viewer.sync()                     # call after releasing the lock
        time.sleep(max(0.0, model.opt.timestep -
                       (time.perf_counter() - wall_start)))

# Alternative, independent use: empty viewer, then load XML interactively.
# mujoco.viewer.launch()
```

Use the same clock for start/end timing. A mutex does not accelerate physics or pause simulated time; it coordinates access between threads. For training, omit this viewer loop and sleep entirely.

### 3.13 `Renderer`, `update_scene`, `render`, and `close`

`Renderer` creates an offscreen camera renderer. `update_scene` prepares geometry from current data; `render` returns an RGB array. This needs a working graphics backend even when no visible window is opened.

```python
renderer = mujoco.Renderer(model, height=240, width=320)
try:
    mujoco.mj_forward(model, data)
    renderer.update_scene(data)            # or camera="camera_name"
    rgb = renderer.render()               # uint8, (240, 320, 3)
finally:
    renderer.close()
```

## 4. MJCF model and actuator API

XML elements are configuration, not Python functions. Their values become fields in `MjModel`. See the [MJCF reference](https://mujoco.readthedocs.io/en/stable/XMLreference.html).

### 4.1 `body`, `joint`, `freejoint`, `geom`, and `site`

`body pos` locates the body's frame relative to its parent. `joint pos` locates the joint pivot inside that body frame; `axis` is also expressed there. `geom` gives collision/visual geometry; `site` is a marker useful for sensors and measurements. A `freejoint` makes a root body movable in 6 DOFs.

```xml
<body name="chassis" pos="0 0 0.5">
  <freejoint name="base"/>
  <geom type="box" size="0.4 0.2 0.05" mass="2"/>
  <site name="imu" pos="0 0 0.05" size="0.01"/>
  <body name="wheel" pos="0 0.25 0">
    <joint name="axle" type="hinge" axis="0 1 0" damping="0.02"/>
    <geom type="cylinder" euler="90 0 0" size="0.15 0.03" mass="0.2"/>
  </body>
</body>
```

This fragment shows a mechanism, not a balanced complete car. Box `size` gives half-lengths; cylinder `size` gives radius and half-length. By default MJCF Euler angles are degrees, while runtime hinge angles are radians. Set `<compiler angle="radian"/>` if you want XML angles in radians.

### 4.2 `motor`: effort control

For a hinge transmission with gear 1, control corresponds to torque in N·m. Other gear ratios/transmissions change the mapping. `ctrlrange` limits command input; `forcerange` limits actuator force. Enable the corresponding flags explicitly in examples.

```xml
<actuator>
  <motor name="wheel_motor" joint="axle" gear="1"
         ctrllimited="true" ctrlrange="-2 2"
         forcelimited="true" forcerange="-2 2"/>
</actuator>
```

```python
data.ctrl[model.actuator("wheel_motor").id] = 0.3  # 0.3 N·m, gear=1
```

### 4.3 `position`: target angle with proportional/derivative feedback

`kp` sets proportional feedback. In the installed MuJoCo version, `kv` on a position actuator adds velocity damping toward zero. For a simple hinge and gear 1, approximately `tau = kp*(q_target-q) - kv*qdot`, before limits and other dynamics.

```xml
<actuator>
  <position name="joint_servo" joint="axle" kp="20" kv="2"
            ctrllimited="true" ctrlrange="-1 1"
            forcelimited="true" forcerange="-2 2"/>
</actuator>
```

```python
data.ctrl[model.actuator("joint_servo").id] = 0.4  # target radians
```

This is not an instant pose assignment. A continuously rotating wheel with a fixed position range eventually reaches that target/range. It does not keep driving indefinitely.

### 4.4 `velocity`: target angular speed

For a hinge and gear 1, approximately `tau = kv*(desired_speed-current_speed)` before limits. Useful for continuous wheel rotation.

```xml
<actuator>
  <velocity name="wheel_speed" joint="axle" kv="0.5"
            ctrllimited="true" ctrlrange="-10 10"
            forcelimited="true" forcerange="-2 2"/>
</actuator>
```

```python
data.ctrl[model.actuator("wheel_speed").id] = 3.0  # rad/s
```

### 4.5 Combined PD plus feedforward using one torque actuator

You can compute position, velocity, and feedforward terms in Python, then send their sum to one motor. This avoids needing one policy output for every controller term. Values below illustrate the formula, not tuned gains for your car.

```python
q = data.qpos[q_index]
dq = data.qvel[v_index]
q_target, dq_target, tau_ff = 0.4, 0.0, 0.0
kp, kd, tau_max = 20.0, 2.0, 2.0
tau = kp * (q_target - q) + kd * (dq_target - dq) + tau_ff
data.ctrl[motor_id] = np.clip(tau, -tau_max, tau_max)  # motor, gear 1
```

Alternatively multiple actuators can contribute forces to one joint; their forces add. Do not unintentionally add damping twice or make a position actuator fight a velocity actuator.

### 4.6 Friction, contact dimensionality, and visual-only marks

`geom friction="sliding torsional rolling"` does not mean XYZ friction. `condim` selects which contact components are modeled: 3 includes tangential sliding, 4 adds torsional friction, 6 also adds rolling friction. Coefficients alone do not activate unused contact dimensions.

```xml
<geom type="cylinder" size="0.15 0.03" mass="0.2"
      friction="0.8 0.005 0.0001" condim="6"/>
<!-- A marker that contributes no collision and no mass. -->
<geom type="box" pos="0.07 0 0.032" size="0.06 0.01 0.001"
      rgba="1 1 1 1" contype="0" conaffinity="0" density="0"/>
```

Use appropriate dimensions and initial heights so wheels do not begin embedded in the floor. Rewards cannot repair an invalid mechanical model.

### 4.7 XML sensors

`jointpos` and `jointvel` expose a joint's scalar state; `gyro` and `accelerometer` read at a site. Their values become entries in `data.sensordata` (section 3.11).

```xml
<sensor>
  <jointpos name="wheel_angle" joint="axle"/>
  <jointvel name="wheel_rate" joint="axle"/>
  <gyro name="imu_gyro" site="imu"/>
  <accelerometer name="imu_accel" site="imu"/>
</sensor>
```

## 5. MuJoCo Warp API

Standalone MJWarp uses Warp arrays, usually on CUDA. These functions were signature-checked against the learning environment. Reference: [MJWarp API](https://mujoco.readthedocs.io/en/stable/mjwarp/api.html). A successful import alone does not prove every model feature works on the backend.

### 5.1 `wp.init`, `wp.ScopedDevice`, `mjw.put_model`, `make_data`, and `put_data`

`init` discovers devices; `ScopedDevice` selects where allocations/operations occur. `put_model` converts a CPU model. `make_data` creates a batch of simulation data. `put_data` initializes a batch from existing CPU state instead.

```python
import warp as wp
import mujoco_warp as mjw

wp.init()
with wp.ScopedDevice("cuda:0"):
    gpu_model = mjw.put_model(model)
    gpu_data = mjw.make_data(model, nworld=128)
    # Alternative, when data contains a desired initial state:
    # gpu_data = mjw.put_data(model, data, nworld=128)
```

`nworld=128` means 128 independent copies, not 128 cars physically interacting in one world. Contact/constraint capacity parameters such as `nconmax` and `njmax` affect allocation; tune them for the model rather than copying Cartpole settings.

### 5.2 `gpu_data.qpos`, `qvel`, and `ctrl`

These Warp arrays have a leading world dimension. `.numpy()` downloads to CPU; it is convenient for diagnosis but can synchronize and slow a training loop.

```python
print(gpu_data.qpos.shape)   # (N, model.nq)
print(gpu_data.qvel.shape)   # (N, model.nv)
print(gpu_data.ctrl.shape)   # (N, model.nu)
q_cpu = gpu_data.qpos.numpy()  # diagnostic download, not a live GPU view
```

### 5.3 `wp.array` and `wp.copy`

`wp.array` allocates/uploads a typed array. `wp.copy(destination, source)` copies values into existing storage. Reuse buffers rather than allocating new GPU arrays every simulation step.

```python
with wp.ScopedDevice("cuda:0"):
    commands = np.zeros(gpu_data.ctrl.shape, dtype=np.float32)
    command_buffer = wp.array(commands, dtype=wp.float32)
    wp.copy(gpu_data.ctrl, command_buffer)
```

### 5.4 `wp.to_torch` and `wp.from_torch`

These provide shared-memory interoperability for compatible arrays/tensors. Keep owners alive, use compatible devices/dtypes, and coordinate CUDA streams when departing from default stream behavior.

```python
import torch
ctrl_view = wp.to_torch(gpu_data.ctrl)
with torch.no_grad():
    ctrl_view.zero_()
    ctrl_view[:, motor_id] = 1.0

actions_t = torch.zeros(gpu_data.ctrl.shape, device="cuda:0", dtype=torch.float32)
actions_w = wp.from_torch(actions_t, dtype=wp.float32)
wp.copy(gpu_data.ctrl, actions_w)
```

### 5.5 `mjw.step`, `mjw.forward`, and `mjw.reset_data`

`step` advances all worlds. `forward` refreshes derived physics without advancing time. `reset_data` resets all worlds or those selected by a boolean Warp mask. Reset task-specific observation histories and rewards separately.

```python
mjw.step(gpu_model, gpu_data)
mjw.forward(gpu_model, gpu_data)

with wp.ScopedDevice("cuda:0"):
    mask_np = np.zeros(gpu_data.qpos.shape[0], dtype=bool)
    mask_np[0] = True
    reset_mask = wp.array(mask_np, dtype=wp.bool)
    mjw.reset_data(gpu_model, gpu_data, reset=reset_mask)
    mjw.forward(gpu_model, gpu_data)
```

### 5.6 `mjw.get_data_into`: copying one world for CPU inspection

This fills an existing CPU `MjData` from a selected GPU world. It does not advance physics and does not display all worlds automatically. Keep the CPU model compatible with the GPU model.

```python
with viewer.lock():                       # fragment: active passive viewer
    mjw.get_data_into(data, model, gpu_data, world_id=0)
    mujoco.mj_forward(model, data)
viewer.sync()
```

Copy at display intervals, not necessarily each physics step. For headless learning, keep observations, policy inference, rewards, and resets on-device when possible.

### 5.7 CUDA graphs: `ScopedCapture`, `capture_launch`, `synchronize`

Capture records GPU work for lower launch overhead; replay launches it again. Warm up first so compilation/allocation is complete. Keep referenced array storage and shapes stable. `synchronize` is needed for accurate GPU wall-time measurement, not after every normal step.

```python
with wp.ScopedDevice("cuda:0"):
    mjw.step(gpu_model, gpu_data)  # warm-up; this also advances simulation
    wp.synchronize()
    with wp.ScopedCapture() as capture:
        mjw.step(gpu_model, gpu_data)
    physics_graph = capture.graph
    for _ in range(100):
        wp.capture_launch(physics_graph)
    wp.synchronize()
```

This graph contains physics only. It does not automatically capture the policy/reward/reset loop. Benchmark total samples per second after warm-up; one tiny world can run faster on CPU.

## 6. Gymnasium API and a complete custom environment

The following is the single-environment contract. Batched SB3/Isaac Lab wrappers have different return conventions; section 7.8 compares them. Reference: [Gymnasium custom environments](https://gymnasium.farama.org/introduction/create_custom_env/).

### 6.1 `gym.Env`, `spaces.Box`, `action_space`, and `observation_space`

Subclass `gymnasium.Env`. The spaces describe what your methods actually accept/return; they do not resize or convert your arrays for you. `Box` describes continuous arrays with a shape, dtype, and bounds.

```python
import gymnasium as gym
from gymnasium import spaces
import numpy as np

class ExampleEnv(gym.Env):
    def __init__(self):
        self.action_space = spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Box(
            -np.inf, np.inf, shape=(8,), dtype=np.float32
        )
```

`action_space.sample()` produces a random valid action. `observation_space.contains(obs)` checks membership; it does not generate a meaningful sensor reading.

```python
env = ExampleEnv()  # space demonstration only; reset/step not implemented here
a = env.action_space.sample()
assert env.observation_space.contains(np.zeros(8, dtype=np.float32))
```

For images, a common raw shape is `(height, width, channels)`, such as `(84, 84, 3)`, with uint8 values 0–255. A vector of 35 measurements should have shape `(35,)`; it is not an image.

### 6.2 `reset(seed=..., options=...)`, `super().reset`, and `np_random`

`reset` starts an episode and returns `(observation, info)`. Call `super().reset(seed=seed)` to establish Gymnasium's generator. Use `self.np_random` for reproducible task randomization. Normally seed once, then call subsequent resets without reusing the identical seed.

```python
# Fragment: method inside a concrete environment class.
def reset(self, *, seed=None, options=None):
    super().reset(seed=seed)
    mujoco.mj_resetData(self.model, self.data)
    self.target = self.np_random.uniform(-1.0, 1.0)
    mujoco.mj_forward(self.model, self.data)
    return self._get_obs(), {"target": float(self.target)}
```

`options` is an optional caller-supplied dictionary for choices such as an evaluation scenario. It is not automatically interpreted by Gymnasium.

### 6.3 `step(action)`: five return values

One call should perform one **policy transition**, which may contain several physics steps. It must return; an unconditional game loop inside `step()` prevents the learner from receiving the transition.

```python
obs, reward, terminated, truncated, info = env.step(action)
done = terminated or truncated
```

`terminated` means the task reached a terminal state (failure or defined success). `truncated` means an external cutoff such as a time limit. PPO uses this distinction when estimating future value; do not arbitrarily collapse it inside a raw Gymnasium environment.

### 6.4 `render_mode`, `render`, `close`, and `TimeLimit`

Choose rendering when constructing an environment. `render()` displays or returns an image according to its mode. `close()` releases resources. `TimeLimit` can impose an episode length when the environment does not implement its own cutoff.

```python
# MotorTargetEnv is defined in the complete example below.
env = MotorTargetEnv(render_mode="human")
obs, info = env.reset(seed=7)
env.render()
env.close()

# Alternative for an env without its own step-count truncation:
# env = gym.wrappers.TimeLimit(MyEnv(), max_episode_steps=500)
```

### 6.5 `gym.register`, `gym.make`, and `check_env`

`register` maps an ID to a class; the registration module must be imported in the process using it. `make` constructs a registered environment. SB3's `check_env` checks return types, spaces, and other compatibility details; it does not prove the reward or physics makes sense.

```python
gym.register(id="MotorTarget-v0", entry_point="motor_env:MotorTargetEnv")
env = gym.make("MotorTarget-v0")

from stable_baselines3.common.env_checker import check_env
check_env(MotorTargetEnv(), warn=True)
```

### 6.6 Complete file: `motor_env.py`

This self-contained MuJoCo environment controls one hinged rotor toward a random angle. It avoids external assets so you can learn the interface before replacing the model with your car. The target is included in the observation so the policy knows what it must achieve.

The observation order is `[sin(angle), cos(angle), angular_speed, target_angle]`, the action is normalized torque, physics runs at 100 Hz, and policy control runs at 50 Hz. Success in `info` is an instantaneous diagnostic, not proof of stable tracking over an entire episode.

```python
# complete-file: motor_env.py
import gymnasium as gym
from gymnasium import spaces
import mujoco
import numpy as np

XML = """
<mujoco model="motor_target">
  <compiler angle="radian"/>
  <option timestep="0.01" gravity="0 0 0" integrator="implicitfast"/>
  <worldbody>
    <light pos="0 0 3"/>
    <body name="rotor" pos="0 0 0.5">
      <joint name="hinge" type="hinge" axis="0 0 1" damping="0.1"
             limited="true" range="-1.5 1.5"/>
      <geom type="capsule" fromto="0 0 0 0.3 0 0" size="0.025"
            mass="0.2" rgba="1 0.7 0 1"/>
      <site name="tip" pos="0.3 0 0" size="0.02"/>
    </body>
  </worldbody>
  <actuator>
    <motor name="motor" joint="hinge" gear="1"
           ctrllimited="true" ctrlrange="-0.5 0.5"/>
  </actuator>
</mujoco>
"""

class MotorTargetEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    def __init__(self, render_mode=None):
        if render_mode not in (None, "human", "rgb_array"):
            raise ValueError("Unsupported render mode")
        self.render_mode = render_mode
        self.model = mujoco.MjModel.from_xml_string(XML)
        self.data = mujoco.MjData(self.model)
        self.frame_skip = 2
        self.control_dt = self.model.opt.timestep * self.frame_skip
        self.max_steps = 250
        self.action_space = spaces.Box(-1.0, 1.0, (1,), np.float32)
        self.observation_space = spaces.Box(-np.inf, np.inf, (4,), np.float32)
        self.target = 0.0
        self.steps = 0
        self._viewer = None
        self._renderer = None

    def _get_obs(self):
        q = float(self.data.qpos[0])
        dq = float(self.data.qvel[0])
        return np.array([np.sin(q), np.cos(q), dq, self.target], dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[0] = self.np_random.uniform(-0.5, 0.5)
        self.target = float(self.np_random.uniform(-1.0, 1.0))
        if options is not None and "target" in options:
            self.target = float(np.clip(options["target"], -1.0, 1.0))
        self.steps = 0
        mujoco.mj_forward(self.model, self.data)
        if self.render_mode == "human":
            self.render()
        return self._get_obs(), {"target": self.target}

    def step(self, action):
        action = np.asarray(action, dtype=np.float32)
        if action.shape != (1,) or not np.isfinite(action).all():
            raise ValueError("Expected one finite normalized action")
        a = float(np.clip(action[0], -1.0, 1.0))
        self.data.ctrl[0] = 0.5 * a
        mujoco.mj_step(self.model, self.data, nstep=self.frame_skip)
        mujoco.mj_forward(self.model, self.data)
        self.steps += 1
        q = float(self.data.qpos[0])
        dq = float(self.data.qvel[0])
        error = q - self.target
        reward = float(-(error**2 + 0.01 * dq**2 + 0.001 * a**2))
        terminated = bool(abs(q) > 1.45)
        truncated = bool(self.steps >= self.max_steps)
        info = {"angle_error": abs(error),
                "is_success": bool(abs(error) < 0.1 and abs(dq) < 0.1)}
        if self.render_mode == "human":
            self.render()
        return self._get_obs(), reward, terminated, truncated, info

    def render(self):
        if self.render_mode == "human":
            if self._viewer is None:
                from mujoco import viewer as mj_viewer
                self._viewer = mj_viewer.launch_passive(self.model, self.data)
            if self._viewer.is_running():
                self._viewer.sync()
            return None
        if self.render_mode == "rgb_array":
            if self._renderer is None:
                self._renderer = mujoco.Renderer(self.model, height=240, width=320)
            self._renderer.update_scene(self.data)
            return self._renderer.render().copy()
        return None

    def close(self):
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
```

To adapt this to your car, change the model, joint/actuator lookup, action mapping, observation construction, reward, failure conditions, and initial state together. Keep the same Gymnasium return contract. For a quadruped, joint states alone usually omit important base orientation and velocity information.

## 7. Stable-Baselines3 PPO training and evaluation

This section applies to an ordinary Gymnasium environment such as `MotorTargetEnv`. It does not make MuJoCo physics run on the GPU just because PPO uses `device="cuda"`.

### 7.1 `PPO` constructor and main parameters

`PPO("MlpPolicy", env, ...)` creates an actor/critic model for vector observations. It has not learned yet.

```python
from stable_baselines3 import PPO
model = PPO(
    "MlpPolicy", env, device="cpu", learning_rate=3e-4,
    n_steps=1024, batch_size=256, n_epochs=10,
    gamma=0.99, gae_lambda=0.95, clip_range=0.2,
    ent_coef=0.0, seed=42, verbose=1, tensorboard_log="logs/sb3"
)
```

| Parameter | Meaning | Small example |
|---|---|---|
| `device` | Device for policy computation | `device="cpu"` |
| `learning_rate` | Optimizer update size | `learning_rate=3e-4` |
| `n_steps` | Rollout transitions per environment before an update | `n_steps=1024` |
| `batch_size` | Minibatch size during updates | `batch_size=256` |
| `n_epochs` | Optimization passes over each collected rollout | `n_epochs=10` |
| `gamma` | Future reward discount | `gamma=0.99` |
| `gae_lambda` | Advantage estimation tradeoff | `gae_lambda=0.95` |
| `clip_range` | PPO probability-ratio clipping parameter | `clip_range=0.2` |
| `ent_coef` | Entropy bonus coefficient | `ent_coef=0.01` |
| `policy_kwargs` | Network customization | `policy_kwargs={"net_arch": [64, 64]}` |
| `tensorboard_log` | Training-log directory | `tensorboard_log="logs/sb3"` |
| `seed` | Random seed for the run | `seed=42` |

Rollout size is `n_steps * n_envs`. `total_timesteps` can be exceeded to finish a rollout. For a small vector-input network and CPU simulator, start on CPU and measure; GPU is especially useful when simulation and large batches also remain on-device. [PPO reference](https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html)

### 7.2 `learn`, `num_timesteps`, and continuing training

`learn()` collects experience and updates parameters. `num_timesteps` tracks collected environment transitions. `reset_num_timesteps=False` preserves that counter across calls; it does **not** load an earlier checkpoint into a newly created model.

```python
model.learn(total_timesteps=100_000, tb_log_name="motor")
print(model.num_timesteps)
model.learn(total_timesteps=100_000, reset_num_timesteps=False, tb_log_name="motor")
```

`model.train()` is not the user-facing complete SB3 training loop. In PyTorch, `network.train()` changes network mode; it does not collect RL experience or run an optimizer automatically.

### 7.3 `save`, `load`, and `predict`

`save` writes a checkpoint. `load` reconstructs it; pass an environment when continuing learning. `predict` returns an action and optional recurrent state. `_` is just a Python variable used when ignoring that second result.

```python
model.save("models/motor_ppo")  # creates motor_ppo.zip
model = PPO.load("models/motor_ppo.zip", env=env, device="cpu")
obs, info = env.reset()
action, _ = model.predict(obs, deterministic=True)
obs, reward, terminated, truncated, info = env.step(action)
```

For evaluation, `deterministic=True` removes action sampling from a standard PPO policy; the environment can still be randomized. A `.zip` does not include your custom environment source or all external normalization state.

### 7.4 `Monitor`, `EvalCallback`, and `CheckpointCallback`

`Monitor` tracks episode returns/lengths. `EvalCallback` periodically evaluates on a separate environment and can save the best checkpoint. `CheckpointCallback` makes periodic recovery checkpoints. Callback frequency counts callback calls, so adjust it for multiple vector environments.

```python
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback

train_env = Monitor(MotorTargetEnv())
eval_env = Monitor(MotorTargetEnv())
evaluation = EvalCallback(eval_env, eval_freq=5000, n_eval_episodes=10,
                          deterministic=True, best_model_save_path="models/best",
                          log_path="logs/eval")
checkpoint = CheckpointCallback(save_freq=5000, save_path="models/checkpoints",
                                name_prefix="motor")
model = PPO("MlpPolicy", train_env, device="cpu")
model.learn(100_000, callback=[evaluation, checkpoint])
```

Never evaluate by stepping the same environment that the learner is currently collecting from. [Callback reference](https://stable-baselines3.readthedocs.io/en/master/guide/callbacks.html)

### 7.5 `evaluate_policy` and manual evaluation

`evaluate_policy` reports mean and standard deviation of episode reward. It does not train. A manual loop is useful for recording task-specific metrics and a fixed test-seed set.

```python
from stable_baselines3.common.evaluation import evaluate_policy
mean_return, std_return = evaluate_policy(
    model, eval_env, n_eval_episodes=20, deterministic=True
)
print(mean_return, std_return)

returns = []
for seed in range(1000, 1010):
    obs, info = eval_env.reset(seed=seed)
    total = 0.0
    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = eval_env.step(action)
        total += reward
        if terminated or truncated:
            break
    returns.append(total)
print("fixed-seed mean:", np.mean(returns))
```

Track speed error, fall rate, distance travelled, torque, and success rate as relevant. A visually pleasant episode alone is not a generalization test; neither is a high reward if the reward has a loophole.

### 7.6 Complete file: `train_motor.py`

Save next to `motor_env.py`. This trains, saves, closes training resources, reloads, and evaluates. The training budget is illustrative; this file does not promise a converged policy.

```python
# complete-file: train_motor.py
from pathlib import Path
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy
from motor_env import MotorTargetEnv

def main():
    root = Path(__file__).resolve().parent
    output = root / "motor_run"
    output.mkdir(exist_ok=True)
    checked = MotorTargetEnv()
    check_env(checked, warn=True)
    checked.close()

    train_env = Monitor(MotorTargetEnv())
    try:
        model = PPO("MlpPolicy", train_env, device="cpu", seed=42,
                    n_steps=1024, batch_size=256, verbose=1,
                    tensorboard_log=str(output / "logs"))
        model.learn(total_timesteps=20_000)
        model.save(str(output / "policy"))
    finally:
        train_env.close()

    test_env = Monitor(MotorTargetEnv())
    try:
        model = PPO.load(str(output / "policy.zip"), device="cpu")
        mean, std = evaluate_policy(model, test_env, n_eval_episodes=10,
                                    deterministic=True)
        print(f"Evaluation return: {mean:.2f} +/- {std:.2f}")
    finally:
        test_env.close()

if __name__ == "__main__":
    main()
```

For visible playback, use a new `MotorTargetEnv(render_mode="human")`, run the manual evaluation loop, and sleep for `control_dt` minus computation time. Keep graphics out of the training loop.

### 7.7 `DummyVecEnv`, `SubprocVecEnv`, and `VecNormalize`

`DummyVecEnv` batches environments in one process; `SubprocVecEnv` uses processes. On Windows, construct subprocess environments under `if __name__ == "__main__":`. These wrappers do not turn CPU MuJoCo into MJWarp.

```python
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize

def make_env():
    return Monitor(MotorTargetEnv())

if __name__ == "__main__":
    vec = SubprocVecEnv([make_env for _ in range(4)])
    # Or: vec = DummyVecEnv([make_env for _ in range(4)])
    norm = VecNormalize(vec, norm_obs=True, norm_reward=True, clip_obs=10.0)
    model = PPO("MlpPolicy", norm, device="cpu")
    model.learn(20_000)
    model.save("policy")
    norm.save("normalization.pkl")
    norm.close()
```

`VecNormalize.load` restores running statistics. Set `training=False` to freeze them during evaluation and `norm_reward=False` to report raw rewards. `normalize_obs` transforms raw observations with saved statistics for a separate inference path; do not normalize twice.

```python
base = DummyVecEnv([make_env])
eval_norm = VecNormalize.load("normalization.pkl", base)
eval_norm.training = False
eval_norm.norm_reward = False
model = PPO.load("policy.zip", env=eval_norm)
mean, std = evaluate_policy(model, eval_norm, n_eval_episodes=20)
eval_norm.close()
```

### 7.8 Reset and step signatures are wrapper-specific

| Interface | Reset/get observation example | Step example |
|---|---|---|
| Raw Gymnasium | `obs, info = env.reset()` | `obs, r, terminated, truncated, info = env.step(a)` |
| SB3 VecEnv | `obs = vec.reset()` | `obs, rewards, dones, infos = vec.step(actions)` |
| Isaac Lab raw DirectRLEnv | `obs_dict, extras = env.reset()` | `obs_dict, rewards, terminated, truncated, extras = env.step(actions)` |
| Local RSL-RL wrapper | `obs_dict = wrapped.get_observations()` | `obs_dict, rewards, dones, extras = wrapped.step(actions)` |

SB3 vector environments automatically reset completed worlds. Their returned observation for a completed world is the next episode's initial observation; the previous final one is in `infos[i]["terminal_observation"]`. Do not feed that reset observation to a custom bootstrap calculation as if it were the final state. [Vector-environment reference](https://stable-baselines3.readthedocs.io/en/master/guide/vec_envs.html)

### 7.9 TensorBoard command

TensorBoard reads event files produced by training; starting its server does not create missing logs. Use the same absolute log directory as training. TensorFlow is not required for ordinary SB3/PyTorch scalar logs.

```powershell
& 'C:\Users\jiaxi\OneDrive\Documentos\GitHub\Littlemoomoo-Python-Learning\.venv\Scripts\python.exe' -m tensorboard.main --logdir 'C:\path\to\motor_run\logs'
```

Open the localhost URL printed by the command, usually `http://localhost:6006`. Stop the server with Ctrl+C. Episode reward and evaluation performance matter more than any single policy/value loss curve.

## 8. Isaac Lab setup, assets, and simulation

Examples in sections 8–12 target the local source snapshot in section 1. They are not generic Isaac Lab 2.x examples. The relevant public guide is [Creating a Direct Workflow RL Environment](https://isaac-sim.github.io/IsaacLab/develop/source/how-to/create_direct_rl_env.html); exact signatures were also inspected in the installed source.

### 8.1 `configclass`, `replace`, and `to_dict`

`configclass` defines structured configuration classes. `replace` makes a configuration variant without modifying the source preset. `to_dict` converts configuration to plain dictionaries, including for RSL-RL.

```python
from isaaclab.utils import configclass, replace, to_dict
from isaaclab_assets.robots.cartpole import CARTPOLE_CFG

robot_cfg = replace(CARTPOLE_CFG, prim_path="{ENV_REGEX_NS}/Robot")
settings = to_dict(robot_cfg)
print(settings["prim_path"])
```

### 8.2 `SimulationCfg`, `NewtonCfg`, and `MJWarpSolverCfg`

`SimulationCfg` sets device and physics timing. `NewtonCfg` selects Newton and its execution settings. `MJWarpSolverCfg` selects/configures the MuJoCo Warp solver within Newton.

```python
from isaaclab.sim import SimulationCfg
from isaaclab_newton.physics import NewtonCfg, MJWarpSolverCfg

sim_cfg = SimulationCfg(
    dt=0.002, device="cuda:0", render_interval=10,
    physics=NewtonCfg(
        solver_cfg=MJWarpSolverCfg(integrator="implicitfast"),
        num_substeps=1, use_cuda_graph=True,
    ),
)
```

| Field | Meaning | Example |
|---|---|---|
| `SimulationCfg.dt` | Simulation step duration | `sim_cfg.dt = 0.002` |
| `device` | Simulation device | `sim_cfg.device = "cuda:0"` |
| `render_interval` | Rendering cadence in simulation steps | `sim_cfg.render_interval = 10` |
| `physics` | Physics backend configuration | `sim_cfg.physics = NewtonCfg()` |
| `NewtonCfg.solver_cfg` | Solver within Newton | `MJWarpSolverCfg(integrator="implicitfast")` |
| `num_substeps` | Internal subdivisions of a simulation step | `NewtonCfg(num_substeps=1)` |
| `use_cuda_graph` | Whether backend uses CUDA graphs | `NewtonCfg(use_cuda_graph=True)` |
| `MJWarpSolverCfg.nconmax` | Contact capacity configuration | `MJWarpSolverCfg(nconmax=64)` |
| `MJWarpSolverCfg.njmax` | Constraint capacity configuration | `MJWarpSolverCfg(njmax=256)` |

The capacity examples are illustrative, not tuned recommendations. Newton subdivisions and RL decimation are different: a policy still advances by `sim.dt * decimation` per action. See the [Newton backend documentation](https://isaac-sim.github.io/IsaacLab/develop/source/concepts/physics-backends/newton.html).

### 8.3 `add_launcher_args` and `launch_simulation`

`add_launcher_args(parser)` adds supported runtime/device/viewer options to an argument parser. `launch_simulation(cfg, args)` starts the runtime required by the config and closes it when the context exits. The current interface can launch Newton without a full Isaac Sim GUI.

```python
import argparse
from isaaclab.app import add_launcher_args, launch_simulation
from isaaclab.sim import SimulationContext

parser = argparse.ArgumentParser()
add_launcher_args(parser)
args = parser.parse_args()
with launch_simulation(sim_cfg, args):
    sim = SimulationContext(sim_cfg)
    # Construct a scene before reset/step in a real simulation.
```

`with` calls the context manager's entry/exit logic. It is a resource-lifetime boundary, not a separate thread or a simulation loop. Older `AppLauncher` examples need version adaptation; the complete example in section 12 uses the current launcher.

### 8.4 `MjcfConverterCfg`, `MjcfConverter`, and `usd_path`

The converter prepares an Isaac Lab asset from MJCF; it is not an instruction to start stepping your old `MjData`. `asset_path` is input XML, `usd_dir` and `usd_file_name` select output, and `fix_base=False` preserves a mobile root. Conversion requires the matching importer dependencies, which are separate from the physics runtime.

```python
# Fragment: needs importer dependencies and an actual input file.
from isaaclab.sim.converters import MjcfConverter, MjcfConverterCfg
from pathlib import Path

source = Path("car.xml").resolve()
conversion_cfg = MjcfConverterCfg(
    asset_path=str(source),
    usd_dir=str(source.parent / "converted_car"),
    usd_file_name="car.usd", fix_base=False,
    import_physics_scene=False,
    run_asset_transformer=True,
    run_multi_physics_conversion=True,
)
converter = MjcfConverter(conversion_cfg)
print(converter.usd_path)
```

`run_asset_transformer` organizes the asset; `run_multi_physics_conversion` prepares compatible backend properties. Import the **robot**, then author the floor/environment separately; importing your old car XML's ground plane and also spawning a floor can create duplicate ground geometry. Recheck inertia, names, axes, collisions, and actuator behavior after conversion. [Asset preparation guide](https://isaac-sim.github.io/IsaacLab/develop/source/how-to/prepare_asset_for_newton.html)

### 8.5 `ArticulationCfg`, `UsdFileCfg`, and `InitialStateCfg`

An articulation is a connected robot with joints. `UsdFileCfg` describes the asset to spawn. `ArticulationCfg.InitialStateCfg` specifies initial base pose and joint state; `joint_pos`/`joint_vel` can select joints by name patterns.

```python
from isaaclab.assets import ArticulationCfg
import isaaclab.sim as sim_utils

car_cfg = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/Robot",
    spawn=sim_utils.UsdFileCfg(usd_path=str(Path("converted_car/car.usd").resolve())),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.5), rot=(0.0, 0.0, 0.0, 1.0),
        joint_pos={".*": 0.0}, joint_vel={".*": 0.0},
    ),
    actuators={},  # supply the actuator configuration in the next entry
)
```

`prim_path` identifies the robot in the scene. `{ENV_REGEX_NS}` lets the scene instantiate one robot per environment. It is not a MuJoCo object name. Height `0.5` is only an example and must match the imported car geometry.

### 8.6 `ImplicitActuatorCfg` and `IdealPDActuatorCfg`

`ImplicitActuatorCfg` uses a solver-side drive. `IdealPDActuatorCfg` computes PD effort explicitly. `stiffness` is proportional gain; `damping` is derivative gain. Joint selection is through `joint_names_expr`.

```python
from isaaclab.actuators import ImplicitActuatorCfg, IdealPDActuatorCfg

car_cfg.actuators = {
    "wheels": ImplicitActuatorCfg(
        joint_names_expr=[".*_axle"], stiffness=0.0, damping=0.5,
        joint_effort_limit=2.0, actuator_velocity_limit=10.0,
    )
}

# Alternative illustration for a position-controlled robot:
leg_drive = IdealPDActuatorCfg(
    joint_names_expr=[".*_hip", ".*_knee"],
    stiffness=20.0, damping=0.5,
    actuator_effort_limit=3.0, joint_effort_limit=3.0,
)
```

With wheel stiffness zero and nonzero damping, a velocity target generates effort from speed error. For a position drive, use nonzero stiffness. The numeric gains and limits here are examples, not your hardware ratings.

`actuator_effort_limit` clips explicit actuator-model output; `joint_effort_limit` is the solver effort limit. `actuator_velocity_limit` describes the motor's speed capability and can feed model/termination logic; it is not a guaranteed hard speed clamp. The MJWarp backend does not enforce every solver limit supported by other backends. [Actuator reference](https://isaac-sim.github.io/IsaacLab/develop/source/api/lab/isaaclab.actuators.html)

### 8.7 `InteractiveSceneCfg`, `AssetBaseCfg`, `GroundPlaneCfg`, and `InteractiveScene`

`InteractiveSceneCfg` declares assets and replication settings. `AssetBaseCfg` configures scene items such as the ground or a light. `InteractiveScene` constructs/manages those runtime objects. In the local DirectRLEnv implementation, the configured scene is built by the base class.

```python
from isaaclab.assets import AssetBaseCfg
from isaaclab.scene import InteractiveSceneCfg, InteractiveScene

@configclass
class CarSceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(prim_path="/World/ground",
                          spawn=sim_utils.GroundPlaneCfg())
    robot: ArticulationCfg = car_cfg

scene_cfg = CarSceneCfg(num_envs=128, env_spacing=4.0)
# Fragment: create inside an initialized standalone simulation runtime.
# scene = InteractiveScene(scene_cfg)
```

`num_envs` controls parallel worlds; `env_spacing` places their origins apart in the scene. `scene["robot"]` accesses the asset named `robot` in the configuration. `scene.env_origins` supplies world offsets for resetting local poses.

### 8.8 Manual Isaac Lab simulation: `reset`, `write_data_to_sim`, `step`, `update`

`SimulationContext.reset()` initializes/resets the simulation context. `scene.write_data_to_sim()` flushes buffered commands. `sim.step()` advances physics. `scene.update(dt)` updates scene buffers/timestamps. `scene.reset()` resets scene-managed state; it does not define your task's random initial poses by itself.

```python
# Fragment: sim and scene have already been constructed.
sim.reset()
scene.reset()
robot = scene["robot"]
targets = torch.zeros_like(robot.data.joint_vel.torch)
for _ in range(100):
    robot.actuators.command.set_velocity_index(value=targets)
    scene.write_data_to_sim()
    sim.step()
    scene.update(sim.get_physics_dt())
```

`sim.get_physics_dt()` returns the configured simulation step duration. Do not add this stepping loop inside `_apply_action()` of DirectRLEnv: its base `step()` already performs simulation and scene updates.

## 9. Isaac Lab state, control, and reset APIs

Here `robot` is an initialized `Articulation` accessed through the scene. `N` is the batch size. All arrays must use compatible devices/dtypes. Public data arrays in this checkout are `ProxyArray` objects; `.torch` exposes their PyTorch view. Use `.clone()` when you need an independent snapshot.

### 9.1 `find_joints`, joint names, and stable policy ordering

`find_joints` resolves names/patterns and returns IDs and matched names. `preserve_order=True` preserves the requested name ordering, which is essential for deployment.

```python
wheel_names = ["Fleft_axle", "Rleft_axle", "Fright_axle", "Rright_axle"]
ids, names = robot.find_joints(wheel_names, preserve_order=True)
wheel_ids = torch.tensor(ids, dtype=torch.long, device=robot.device)
assert names == wheel_names  # verify after importing the model
```

If importing changed names, inspect `robot.joint_names`, then update the explicit list. Do not infer motor ordering from alphabetical sorting.

```python
print(robot.joint_names)
```

### 9.2 `Articulation.data` members

| Member accessed through `robot.data` | Meaning | Example |
|---|---|---|
| `joint_pos` | Joint position, `(N,J)` | `q = robot.data.joint_pos.torch` |
| `joint_vel` | Joint velocity, `(N,J)` | `dq = robot.data.joint_vel.torch` |
| `joint_acc` | Joint acceleration | `ddq = robot.data.joint_acc.torch` |
| `default_joint_pos` | Configured default joint positions | `q0 = robot.data.default_joint_pos.torch.clone()` |
| `default_joint_vel` | Configured default joint velocities | `dq0 = robot.data.default_joint_vel.torch.clone()` |
| `default_root_pose` | Default root link XYZ + xyzw pose, environment-local | `pose = robot.data.default_root_pose.torch.clone()` |
| `default_root_vel` | Default root velocity | `vel = robot.data.default_root_vel.torch.clone()` |
| `root_link_pos_w` | Root link origin position in world | `p = robot.data.root_link_pos_w.torch` |
| `root_link_quat_w` | Root link orientation, xyzw | `qbase = robot.data.root_link_quat_w.torch` |
| `root_com_pos_w` | Root body's COM world position | `pcom = robot.data.root_com_pos_w.torch` |
| `root_com_lin_vel_w` | Root COM linear velocity in world axes | `v = robot.data.root_com_lin_vel_w.torch` |
| `root_com_lin_vel_b` | Root COM linear velocity in body axes | `v = robot.data.root_com_lin_vel_b.torch` |
| `root_link_lin_vel_b` | Root link origin linear velocity in body axes | `v = robot.data.root_link_lin_vel_b.torch` |
| `root_link_ang_vel_b` | Root angular velocity in body axes | `omega = robot.data.root_link_ang_vel_b.torch` |
| `projected_gravity_b` | Unit gravity direction expressed in body axes | `g = robot.data.projected_gravity_b.torch` |
| `soft_joint_pos_limits` | Task-friendly lower/upper joint bounds | `limits = robot.data.soft_joint_pos_limits.torch` |

The root body's COM is not necessarily the whole quadruped's combined COM. Avoid ambiguous aliases such as `root_state_w` when the reference point matters; use explicit link/COM members.

### 9.3 `actuators.command.set_position_index`

Writes desired joint positions to command buffers. It does not teleport joint state. `value` has shape `(selected_envs, selected_joints)` with the default `full_data=False`.

```python
q_target = robot.data.default_joint_pos.torch[:, joint_ids] + 0.25 * actions
robot.actuators.command.set_position_index(value=q_target, joint_ids=joint_ids)
```

This is a common quadruped action map: one normalized policy action becomes an offset from a standing pose. `joint_ids` must match the policy action order.

### 9.4 `actuators.command.set_velocity_index`

Writes desired joint speeds. For two car actions, duplicate each side's target across its front/rear wheel. Check axis signs before training.

```python
# Fragment: actions shape (N,2), wheel_ids order [FL, RL, FR, RR].
side_speeds = actions.clamp(-1.0, 1.0) * 8.0  # rad/s
wheel_targets = side_speeds[:, [0, 0, 1, 1]]
robot.actuators.command.set_velocity_index(value=wheel_targets, joint_ids=wheel_ids)
```

If left/right joint axes have opposing definitions, include a fixed sign vector validated with manual commands. A policy should not be required to compensate for an undocumented sign convention.

### 9.5 `actuators.command.set_effort_index` and combined commands

Writes feedforward effort targets (N·m for revolute joints). A drive may also contribute PD effort; zero the appropriate gains if you intend pure torque control.

```python
tau = (2.0 * actions).clamp(-2.0, 2.0)
robot.actuators.command.set_effort_index(value=tau, joint_ids=joint_ids)

# A PD drive can use all three targets together:
robot.actuators.command.set_position_index(value=q_target, joint_ids=joint_ids)
robot.actuators.command.set_velocity_index(value=dq_target, joint_ids=joint_ids)
robot.actuators.command.set_effort_index(value=tau_ff, joint_ids=joint_ids)
```

There is still one physical joint. Position, velocity, and feedforward effort are controller inputs, not necessarily three independent RL outputs.

### 9.6 Compatibility methods used in some current examples

The local source still contains these methods, but marks them deprecated in favor of the command interface above. They accept `target=`, whereas the command interface accepts `value=`.

```python
robot.set_joint_position_target_index(target=q_target, joint_ids=joint_ids)
robot.set_joint_velocity_target_index(target=dq_target, joint_ids=joint_ids)
robot.set_joint_effort_target_index(target=tau_ff, joint_ids=joint_ids)
```

All three set buffers; `write_data_to_sim` applies them during the simulation pipeline. Reading old examples with these names is fine; new code should prefer `robot.actuators.command` for this checkout.

### 9.7 State writes for reset: root pose and root velocity

`write_root_pose_to_sim_index` writes root link pose, effectively repositioning the robot. `write_root_velocity_to_sim_index` writes root COM velocity using linear-then-angular world components in this interface. These are reset/state-edit operations, not actuator commands.

```python
# Fragment: env_ids selects worlds ending an episode.
pose = robot.data.default_root_pose.torch[env_ids].clone()
pose[:, :3] += scene.env_origins[env_ids]
vel = robot.data.default_root_vel.torch[env_ids].clone()
robot.write_root_pose_to_sim_index(root_pose=pose, env_ids=env_ids)
robot.write_root_velocity_to_sim_index(root_velocity=vel, env_ids=env_ids)
```

Always add each environment's origin when converting an environment-local default pose to world coordinates. Otherwise many reset robots can be placed at the same world location.

### 9.8 State writes for reset: joint position and velocity

`write_joint_position_to_sim_index` sets actual joint positions; `write_joint_velocity_to_sim_index` sets actual joint velocities. Their `position`/`velocity` arrays contain only the selected environments/joints.

```python
q = robot.data.default_joint_pos.torch[env_ids].clone()
dq = robot.data.default_joint_vel.torch[env_ids].clone()
q += 0.02 * (2.0 * torch.rand_like(q) - 1.0)
robot.write_joint_position_to_sim_index(position=q, env_ids=env_ids)
robot.write_joint_velocity_to_sim_index(velocity=dq, env_ids=env_ids)
```

Clear action history, command filters, recurrent state, and task accumulators for those same worlds. Do not reset all environments because one falls.

### 9.9 `sample_uniform`, `quat_apply_inverse`, and `convert_quat`

`sample_uniform` creates batched random samples on the requested device. `quat_apply_inverse` rotates world vectors into local coordinates for a quaternion describing local-to-world orientation. `convert_quat` changes component ordering, not the physical rotation.

```python
from isaaclab.utils.math import sample_uniform, quat_apply_inverse, convert_quat
noise = sample_uniform(-0.05, 0.05, q.shape, q.device)
v_b = quat_apply_inverse(robot.data.root_link_quat_w.torch,
                         robot.data.root_link_lin_vel_w.torch)
q_xyzw = convert_quat(q_wxyz, to="xyzw")
```

## 10. Isaac Lab custom environment hooks

### 10.1 `DirectRLEnvCfg` and `DirectRLEnv`

`DirectRLEnvCfg` describes the task and scene. `DirectRLEnv` owns the batched reset/step loop; your subclass provides action processing, observations, rewards, terminations, and resets. A direct workflow is often easiest while learning how these pieces interact.

```python
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg

@configclass
class CarEnvCfg(DirectRLEnvCfg):
    decimation = 10
    episode_length_s = 10.0
    action_space = 2
    observation_space = 15
    state_space = 0
    sim = sim_cfg
    scene = CarSceneCfg(num_envs=128, env_spacing=4.0)
```

The 15 observations below are one proposed car contract: command 2 + linear velocity 3 + angular velocity 3 + gravity direction 3 + wheel speeds 4. They are not a universal required observation size.

| Field/member | Meaning | Example |
|---|---|---|
| `cfg.decimation` | Physics steps per policy action | `cfg.decimation = 10` |
| `cfg.episode_length_s` | Episode duration in simulation seconds | `cfg.episode_length_s = 10.0` |
| `cfg.action_space` | Scalar action dimensions per environment | `cfg.action_space = 2` |
| `cfg.observation_space` | Policy observation dimensions | `cfg.observation_space = 15` |
| `cfg.state_space` | Optional privileged critic-state dimensions; 0 disables | `cfg.state_space = 0` |
| `self.device` | Runtime device | `torch.zeros(2, device=self.device)` |
| `self.num_envs` | Batch size | `torch.zeros(self.num_envs, device=self.device)` |
| `self.step_dt` | One policy transition's simulation duration | `elapsed += self.step_dt` |
| `self.episode_length_buf` | Steps elapsed per environment | `expired = self.episode_length_buf >= self.max_episode_length` |
| `self.max_episode_length` | Episode horizon in policy steps | `print(self.max_episode_length)` |
| `self.extras` | Extra metrics/info dictionary | `self.extras["log"] = {"mean_speed": speed.mean()}` |

### 10.2 `_pre_physics_step(actions)`

Called once per policy step. Clamp/scale actions and save values used over the coming substeps. The following method fragments assume your constructor created `self.robot`, `self.wheel_ids`, `self.commands` with shape `(N,2)`, and `self.actions` with shape `(N,2)`.

```python
def _pre_physics_step(self, actions):
    self.actions = actions.clamp(-1.0, 1.0).clone()
    self.wheel_targets = 8.0 * self.actions[:, [0, 0, 1, 1]]
```

### 10.3 `_apply_action()`

Called on each physics substep. Set actuator commands; the base environment flushes them and steps physics. Do not call `sim.step()` again here.

```python
def _apply_action(self):
    self.robot.actuators.command.set_velocity_index(
        value=self.wheel_targets, joint_ids=self.wheel_ids
    )
```

### 10.4 `_get_observations()`

Returns observation groups, normally including `"policy"`. Preserve column order and dimensions exactly. Use clone when saving past values; do not unintentionally keep aliases to changing simulator state.

```python
def _get_observations(self):
    obs = torch.cat([
        self.commands,                                    # 2: target vx, yaw rate
        self.robot.data.root_link_lin_vel_b.torch,         # 3
        self.robot.data.root_link_ang_vel_b.torch,         # 3
        self.robot.data.projected_gravity_b.torch,         # 3
        self.robot.data.joint_vel.torch[:, self.wheel_ids] # 4
    ], dim=-1)
    return {"policy": obs}  # (N,15)
```

### 10.5 `_get_rewards()`

Returns one scalar per environment. The example tracks commanded forward speed and yaw rate and penalizes large actions. Reward weights and tolerances are design choices, not library magic.

```python
def _get_rewards(self):
    vx = self.robot.data.root_link_lin_vel_b.torch[:, 0]
    wz = self.robot.data.root_link_ang_vel_b.torch[:, 2]
    error = ((vx - self.commands[:, 0]) / 0.5).square()
    error += ((wz - self.commands[:, 1]) / 1.0).square()
    tracking = torch.exp(-error)
    effort_proxy = 0.01 * self.actions.square().sum(dim=-1)
    return (tracking - effort_proxy) * self.step_dt
```

The divisions make velocity errors dimensionless using explicit tolerances. Multiplying a reward rate by `step_dt` makes its cumulative scale less sensitive to control frequency. An action penalty is not an actual energy measurement. Measure torque and joint velocity for physical power estimates.

### 10.6 `_get_dones()`

Returns `(terminated, time_out)` as boolean tensors `(N,)`. For an upright z-up car, projected gravity is near `[0,0,-1]`; the example treats a large tilt as failure.

```python
def _get_dones(self):
    gravity_z = self.robot.data.projected_gravity_b.torch[:, 2]
    overturned = gravity_z > -0.3
    time_out = self.episode_length_buf >= self.max_episode_length
    return overturned, time_out
```

This checks orientation, not whether the car is airborne or stuck. Add task-appropriate criteria deliberately; excessive early termination can prevent useful exploration.

### 10.7 `_reset_idx(env_ids)`

Resets only selected worlds. Call the base implementation for bookkeeping. Resample commands and clear history in addition to restoring physical state.

```python
def _reset_idx(self, env_ids):
    if env_ids is None:
        env_ids = torch.arange(self.num_envs, device=self.device)
    super()._reset_idx(env_ids)
    pose = self.robot.data.default_root_pose.torch[env_ids].clone()
    pose[:, :3] += self.scene.env_origins[env_ids]
    vel = self.robot.data.default_root_vel.torch[env_ids].clone()
    q = self.robot.data.default_joint_pos.torch[env_ids].clone()
    dq = self.robot.data.default_joint_vel.torch[env_ids].clone()
    self.robot.write_root_pose_to_sim_index(root_pose=pose, env_ids=env_ids)
    self.robot.write_root_velocity_to_sim_index(root_velocity=vel, env_ids=env_ids)
    self.robot.write_joint_position_to_sim_index(position=q, env_ids=env_ids)
    self.robot.write_joint_velocity_to_sim_index(velocity=dq, env_ids=env_ids)
    self.actions[env_ids] = 0.0
    self.wheel_targets[env_ids] = 0.0
    self.commands[env_ids] = 2.0 * torch.rand((len(env_ids), 2), device=self.device) - 1.0
```

### 10.8 Registration and construction

The registration specifies an environment implementation and configuration entry points. `disable_env_checker=True` is used for Isaac Lab's batched tensor contract; it is not an excuse to skip shape checks in your code.

```python
# Fragment in my_tasks/__init__.py after implementing these modules/classes.
gym.register(
    id="My-Car-Direct-v0", entry_point="my_tasks.car_env:CarEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "my_tasks.car_env_cfg:CarEnvCfg",
        "rsl_rl_cfg_entry_point": "my_tasks.agents:CarPPORunnerCfg",
    },
)

# In a driver, after importing my_tasks and launching its runtime:
# env = gym.make("My-Car-Direct-v0", cfg=cfg)
```

`resolve_task_config(task_id, agent_entry_point)` resolves registered environment/agent configurations and preset overrides. A registration file must actually be imported; simply creating it in a random folder does not register it globally.

```python
import isaaclab_tasks
from isaaclab_tasks.utils import resolve_task_config
env_cfg, agent_cfg = resolve_task_config("Isaac-Cartpole-Direct", "rsl_rl_cfg_entry_point")
```

## 11. Manager-based environments and sensors

A manager-based environment expresses the same task using configuration terms instead of implementing all direct hooks. Choose one workflow for your first task; you do not need to implement both. The following fragments show where the main manager APIs belong.

### 11.1 `ManagerBasedRLEnvCfg` and `ManagerBasedRLEnv`

The config groups scene, actions, observations, rewards, terminations, and events. The runtime calls their managers in its step/reset pipeline. Start from the installed Cartpole manager config to see every required group wired together.

```python
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab_tasks.core.cartpole.cartpole_manager_env_cfg import CartpoleEnvCfg
cfg = CartpoleEnvCfg()
cfg.scene.num_envs = 64
# Inside launch_simulation(cfg, ...):
# env = ManagerBasedRLEnv(cfg)
```

### 11.2 `SceneEntityCfg`

Selects an asset and its named joints/bodies for a manager term. Managers resolve those names to indices.

```python
from isaaclab.managers import SceneEntityCfg
wheels = SceneEntityCfg("robot", joint_names=[".*_axle"])
```

### 11.3 `JointPositionActionCfg`, `JointVelocityActionCfg`, `JointEffortActionCfg`

These map policy outputs to joint commands with a scale. Position actions can add default joint positions. Velocity and effort actions select the corresponding controller input. Each typically allocates one policy dimension per selected joint; a two-output-to-four-wheel mapping requires a custom action term or the direct implementation in section 10.

```python
from isaaclab.envs import mdp
position_action = mdp.JointPositionActionCfg(
    asset_name="robot", joint_names=[".*_hip", ".*_knee"],
    scale=0.25, use_default_offset=True,
)
velocity_action = mdp.JointVelocityActionCfg(
    asset_name="robot", joint_names=[".*_axle"], scale=8.0,
)
effort_action = mdp.JointEffortActionCfg(
    asset_name="robot", joint_names=[".*_axle"], scale=2.0,
)
```

### 11.4 `ObservationTermCfg` and `ObservationGroupCfg`

A term calls one observation function. A group specifies how terms are combined and whether corruption/noise is enabled. Definition order becomes column order when concatenating, so preserve it for deployment.

```python
from isaaclab.managers import ObservationTermCfg, ObservationGroupCfg

@configclass
class PolicyObsCfg(ObservationGroupCfg):
    q = ObservationTermCfg(func=mdp.joint_pos_rel)
    dq = ObservationTermCfg(func=mdp.joint_vel_rel)
    def __post_init__(self):
        self.concatenate_terms = True
        self.enable_corruption = False
```

`mdp.joint_pos_rel` subtracts default joint positions; `mdp.joint_vel_rel` subtracts default velocities. Their returned shapes depend on the selected asset/joints.

```python
# Fragment: initialized manager-based env with its default asset named robot.
q_rel = mdp.joint_pos_rel(env)
dq_rel = mdp.joint_vel_rel(env)
```

### 11.5 `RewardTermCfg` and custom reward functions

A reward term calls a function returning `(N,)`, then applies its configured weight. In the normal manager workflow, the manager integrates reward terms using the environment step duration; do not also multiply by `step_dt` inside a reward-rate function unless that double scaling is intentional.

```python
from isaaclab.managers import RewardTermCfg

def forward_speed_error(env, target_speed: float):
    speed = env.scene["robot"].data.root_link_lin_vel_b.torch[:, 0]
    return (speed - target_speed).square()

speed_cost = RewardTermCfg(func=forward_speed_error, weight=-1.0,
                           params={"target_speed": 0.5})
```

If target speed varies between episodes, supply it through the task's command state and observation instead of hiding it inside the reward only.

### 11.6 `TerminationTermCfg` and `mdp.time_out`

A termination term is a boolean test. `time_out=True` labels the condition as truncation. `mdp.time_out` tests the environment's configured horizon.

```python
from isaaclab.managers import TerminationTermCfg
timeout = TerminationTermCfg(func=mdp.time_out, time_out=True)

def tipped_over(env):
    return env.scene["robot"].data.projected_gravity_b.torch[:, 2] > -0.3

fall = TerminationTermCfg(func=tipped_over)
```

### 11.7 `EventTermCfg` and `reset_joints_by_offset`

Events run at startup, reset, or configured intervals. `reset_joints_by_offset` perturbs default joint state for selected environments. This is one form of randomization; mass/friction/latency randomization must use backend-supported properties and physically plausible ranges.

```python
from isaaclab.managers import EventTermCfg
reset_joints = EventTermCfg(
    func=mdp.reset_joints_by_offset, mode="reset",
    params={"asset_cfg": SceneEntityCfg("robot"),
            "position_range": (-0.05, 0.05),
            "velocity_range": (-0.1, 0.1)},
)
```

### 11.8 `ContactSensorCfg`, `scene.sensors`, and contact forces

The sensor config selects bodies. `history_length` retains recent measurements; `update_period=0.0` requests each simulation update. `net_normal_forces_w` contains aggregate normal contact force in world axes; use explicit normal/total/friction fields deliberately because backend support varies.

```python
from isaaclab.sensors import ContactSensorCfg
foot_sensor_cfg = ContactSensorCfg(
    prim_path="{ENV_REGEX_NS}/Robot/.*foot", update_period=0.0,
    history_length=3, track_air_time=True,
)
# Add this as a named member of the scene config before construction.
# Fragment after initialization:
sensor = env.scene.sensors["foot_contacts"]
forces = sensor.data.net_normal_forces_w.torch  # (N, selected_bodies, 3)
touching = torch.linalg.vector_norm(forces, dim=-1) > 1.0
```

The `1.0 N` threshold is illustrative. Enabling contact reporting can require asset/backend configuration; a body name pattern must match imported bodies. Foot normal contact does not by itself reveal slipping.

### 11.9 `CameraCfg`, `PinholeCameraCfg`, and `Camera.data.output`

`CameraCfg` specifies image dimensions, requested channels, pose, and renderer. `PinholeCameraCfg` sets optical parameters. A camera renderer must support the requested output and be configured for the chosen backend; copying an RTX camera config does not make it work automatically with every Newton renderer.

```python
from isaaclab.sensors import CameraCfg
camera_cfg = CameraCfg(
    prim_path="{ENV_REGEX_NS}/Robot/Camera", width=84, height=84,
    data_types=["rgb"],
    spawn=sim_utils.PinholeCameraCfg(
        focal_length=24.0, horizontal_aperture=20.955,
        clipping_range=(0.1, 20.0),
    ),
)
# Configure camera_cfg.renderer_cfg using a supported renderer preset.
# Fragment after scene construction and a rendered update:
rgb = env.scene.sensors["camera"].data.output["rgb"].torch
image_batch = rgb[..., :3].permute(0, 3, 1, 2).float() / 255.0
```

This preprocessing assumes uint8 0–255 RGB. Verify the renderer's actual dtype/range before dividing. Preserve training resolution, channels, normalization, and camera calibration during deployment. Start with state observations until basic motion control works.

## 12. Isaac Lab training, playback, and export

### 12.1 `RslRlMLPModelCfg`, `RslRlPpoAlgorithmCfg`, and `RslRlOnPolicyRunnerCfg`

The model config defines actor/critic network layers. The algorithm config defines PPO updates. The runner config sets rollout length, iterations, logging, and checkpoints. In this checkout actor and critic are configured separately; older `RslRlPpoActorCriticCfg` examples are not the API used here.

```python
from isaaclab_rl.rsl_rl import (
    RslRlMLPModelCfg, RslRlPpoAlgorithmCfg, RslRlOnPolicyRunnerCfg,
)
actor_cfg = RslRlMLPModelCfg(
    hidden_dims=[64, 64], activation="elu", obs_normalization=True,
    distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=0.5),
)
# Reuse a complete installed agent config, then adjust it deliberately.
from isaaclab_tasks.core.cartpole.agents.rsl_rl_ppo_cfg import CartpoleDirectPPORunnerCfg
agent_cfg = CartpoleDirectPPORunnerCfg()
agent_cfg.actor = actor_cfg
agent_cfg.num_steps_per_env = 32
agent_cfg.max_iterations = 200
agent_cfg.obs_groups = {"actor": ["policy"], "critic": ["policy"]}
agent_cfg.algorithm.learning_rate = 3e-4
```

`GaussianDistributionCfg.init_std` initializes action exploration noise. `obs_normalization` controls the model's observation normalization. A complete `RslRlPpoAlgorithmCfg` includes parameters beyond learning rate; inheriting the installed complete config avoids missing mandatory fields.

| Runner/algorithm field | Meaning | Example |
|---|---|---|
| `num_steps_per_env` | Rollout length per world | `agent_cfg.num_steps_per_env = 32` |
| `max_iterations` | Number of policy updates requested | `agent_cfg.max_iterations = 200` |
| `obs_groups` | Maps actor/critic to environment observation groups | `agent_cfg.obs_groups = {"actor": ["policy"], "critic": ["policy"]}` |
| `clip_actions` | Wrapper action clipping | `agent_cfg.clip_actions = 1.0` |
| `save_interval` | Checkpoint interval in iterations | `agent_cfg.save_interval = 50` |
| `logger` | Logging integration | `agent_cfg.logger = "tensorboard"` |
| `algorithm.num_learning_epochs` | Update passes per rollout | `agent_cfg.algorithm.num_learning_epochs = 5` |
| `algorithm.num_mini_batches` | Number of minibatches per epoch | `agent_cfg.algorithm.num_mini_batches = 4` |
| `algorithm.clip_param` | PPO ratio clipping | `agent_cfg.algorithm.clip_param = 0.2` |
| `algorithm.gamma` | Discount factor | `agent_cfg.algorithm.gamma = 0.99` |
| `algorithm.lam` | GAE parameter | `agent_cfg.algorithm.lam = 0.95` |

Collected transitions are approximately `num_envs * num_steps_per_env * iterations`. Do not compare 200 Isaac Lab iterations with 200 SB3 timesteps as if they were the same amount of experience.

### 12.2 `RslRlVecEnvWrapper` and `OnPolicyRunner`

The wrapper adapts Isaac Lab's environment contract to RSL-RL and resets it at construction. `OnPolicyRunner` owns experience collection and policy optimization. `to_dict(agent_cfg)` supplies its plain configuration dictionary.

```python
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from rsl_rl.runners import OnPolicyRunner
wrapped = RslRlVecEnvWrapper(env, clip_actions=1.0)
runner = OnPolicyRunner(wrapped, to_dict(agent_cfg),
                         log_dir="logs/my_task", device="cuda:0")
```

### 12.3 Runner `learn`, `save`, and `load`

`learn` performs training iterations. `save` writes a training checkpoint. `load` restores model and training state by default; for a changed optimizer/config, inspect `load_cfg` in the installed version rather than assuming an older `load_optimizer` parameter exists.

```python
runner.learn(num_learning_iterations=200, init_at_random_ep_len=True)
runner.save("logs/my_task/final.pt")
runner.load("logs/my_task/final.pt", map_location="cuda:0")
runner.learn(num_learning_iterations=50, init_at_random_ep_len=False)
```

### 12.4 `get_inference_policy`, wrapper `get_observations`, and `policy.reset`

`get_inference_policy` returns the actor in inference mode. In local RSL-RL 5.5.1, it consumes the wrapper's observation dictionary/TensorDict. `policy.reset(dones)` clears recurrent state for completed worlds; the simple MLP has no useful recurrent memory to clear.

```python
policy = runner.get_inference_policy(device=wrapped.unwrapped.device)
obs = wrapped.get_observations()  # not (obs, extras) in this installed version
with torch.inference_mode():
    for _ in range(500):
        actions = policy(obs)
        obs, rewards, dones, extras = wrapped.step(actions)
        policy.reset(dones)
```

The environment automatically resets completed worlds. Accumulate returns before clearing each world's episode accumulator. Do not use the training loop again just to watch a learned policy.

### 12.5 Current CLI: list, debug, train, resume, play

The following commands use the already-installed executable directly, avoiding implicit dependency updates from a package manager. `uv run isaaclab ...` is also the project's normal interface when run in the configured checkout.

```powershell
Set-Location 'C:\Users\jiaxi\IsaacLab'
.\.venv\Scripts\isaaclab.exe list_envs
.\.venv\Scripts\isaaclab.exe train --rl_library rsl_rl --task Isaac-Cartpole-Direct --help

# Headless by default in this checkout; no viewer selector is supplied.
.\.venv\Scripts\isaaclab.exe train --rl_library rsl_rl --task Isaac-Cartpole-Direct --num_envs 128 --max_iterations 100 physics=newton_mjwarp

# Resume a particular checkpoint, preserving the task's matching config.
.\.venv\Scripts\isaaclab.exe train --rl_library rsl_rl --task Isaac-Cartpole-Direct --checkpoint 'C:\path\to\model_99.pt' --num_envs 128 --max_iterations 50 physics=newton_mjwarp

# Watch a small batch; this does not train.
.\.venv\Scripts\isaaclab.exe play --rl_library rsl_rl --task Isaac-Cartpole-Direct --checkpoint 'C:\path\to\model_99.pt' --num_envs 4 --visualizer newton_gl physics=newton_mjwarp
```

`physics=newton_mjwarp` selects a named backend preset. `--visualizer newton_gl` selects a viewer. Use `random_agent` or `zero_agent` with a registered task before training to check reset, command mapping, and stability. Available viewers/presets depend on installed extras and the task.

```powershell
.\.venv\Scripts\isaaclab.exe random_agent --task Isaac-Cartpole-Direct --num_envs 4 physics=newton_mjwarp
```

Old `--headless` and `scripts/reinforcement_learning/.../train.py` commands are not the interface validated for this source revision. CLI training/playback also saves run metadata/configuration; prefer it for full experiments after your custom task is registered.

### 12.6 Complete file: `isaac_custom_env.py`

This defines a new direct-workflow Cartpole task, using the installed Cartpole scene/asset so there is no invented USD path. All six environment hooks are visible. It is a learning example, not a conversion of your car. Replacing the scene and the six hooks is the route to a different robot.

**Asset prerequisite:** the preset refers to NVIDIA-hosted Cartpole and floor assets, not bundled local USD files. They must be available in the cache or reachable by the runtime. The runtime check for this document stopped because the configured floor asset could not be retrieved; full Isaac Lab training/evaluation below remains unverified on this machine. Its imports, configuration construction, and API signatures were checked. See section 15.1.

```python
# complete-file: isaac_custom_env.py
import torch
from isaaclab.envs import DirectRLEnv
from isaaclab.utils import configclass
from isaaclab_tasks.core.cartpole.cartpole_direct_env_cfg import CartpoleEnvCfg

@configclass
class LearningBalanceCfg(CartpoleEnvCfg):
    action_scale = 10.0
    episode_length_s = 5.0
    action_space = 1
    observation_space = 4
    state_space = 0

class LearningBalanceEnv(DirectRLEnv):
    def __init__(self, cfg, render_mode=None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self.robot = self.scene["cartpole"]
        cart, _ = self.robot.find_joints("slider_to_cart")
        pole, _ = self.robot.find_joints("cart_to_pole")
        self.cart = torch.tensor(cart, device=self.device, dtype=torch.long)
        self.pole = torch.tensor(pole, device=self.device, dtype=torch.long)
        self.actions = torch.zeros((self.num_envs, 1), device=self.device)

    def _pre_physics_step(self, actions):
        self.actions = actions.clamp(-1.0, 1.0).clone()

    def _apply_action(self):
        self.robot.actuators.command.set_effort_index(
            value=self.cfg.action_scale * self.actions, joint_ids=self.cart
        )

    def _get_observations(self):
        q = self.robot.data.joint_pos.torch
        dq = self.robot.data.joint_vel.torch
        obs = torch.cat([q[:, self.cart], q[:, self.pole],
                         dq[:, self.cart], dq[:, self.pole]], dim=-1)
        return {"policy": obs}

    def _get_rewards(self):
        q = self.robot.data.joint_pos.torch
        dq = self.robot.data.joint_vel.torch
        pole_cost = q[:, self.pole].square().sum(dim=-1)
        cart_cost = 0.1 * q[:, self.cart].square().sum(dim=-1)
        motion_cost = 0.01 * dq.square().sum(dim=-1)
        failure_cost = 5.0 * self.reset_terminated.float()
        return (1.0 - pole_cost - cart_cost - motion_cost - failure_cost) * self.step_dt

    def _get_dones(self):
        q = self.robot.data.joint_pos.torch
        cart_out = (q[:, self.cart].abs() > 2.0).any(dim=-1)
        pole_fell = (q[:, self.pole].abs() > 0.8).any(dim=-1)
        time_out = self.episode_length_buf >= self.max_episode_length
        return cart_out | pole_fell, time_out

    def _reset_idx(self, env_ids):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        super()._reset_idx(env_ids)
        q = self.robot.data.default_joint_pos.torch[env_ids].clone()
        dq = self.robot.data.default_joint_vel.torch[env_ids].clone()
        q += 0.1 * (2.0 * torch.rand_like(q) - 1.0)
        pose = self.robot.data.default_root_pose.torch[env_ids].clone()
        pose[:, :3] += self.scene.env_origins[env_ids]
        vel = self.robot.data.default_root_vel.torch[env_ids].clone()
        self.robot.write_root_pose_to_sim_index(root_pose=pose, env_ids=env_ids)
        self.robot.write_root_velocity_to_sim_index(root_velocity=vel, env_ids=env_ids)
        self.robot.write_joint_position_to_sim_index(position=q, env_ids=env_ids)
        self.robot.write_joint_velocity_to_sim_index(velocity=dq, env_ids=env_ids)
        self.actions[env_ids] = 0.0
```

### 12.7 Complete file: `isaac_train_evaluate.py`

Save next to `isaac_custom_env.py`. This directly constructs the custom task, so registration is unnecessary for this particular script. `--checkpoint` switches to evaluation of an existing checkpoint. No extra `learn` call is made in that mode. The explicit observation-group mapping matches current RSL-RL.

```python
# complete-file: isaac_train_evaluate.py
import argparse
from pathlib import Path
import torch
from isaaclab.app import launch_simulation
from isaaclab.utils import to_dict
from isaaclab_newton.physics import NewtonCfg, MJWarpSolverCfg
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab_tasks.core.cartpole.agents.rsl_rl_ppo_cfg import CartpoleDirectPPORunnerCfg
from rsl_rl.runners import OnPolicyRunner
from isaac_custom_env import LearningBalanceCfg, LearningBalanceEnv

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--checkpoint", type=str, default=None)
    args = parser.parse_args()
    output = Path(__file__).resolve().parent / "isaac_balance_run"
    output.mkdir(exist_ok=True)

    cfg = LearningBalanceCfg()
    cfg.scene.num_envs = args.num_envs
    cfg.seed = 42
    cfg.sim.device = "cuda:0"
    cfg.sim.physics = NewtonCfg(
        solver_cfg=MJWarpSolverCfg(integrator="implicitfast"),
        use_cuda_graph=True,
    )
    agent_cfg = CartpoleDirectPPORunnerCfg()
    agent_cfg.device = "cuda:0"
    agent_cfg.obs_groups = {"actor": ["policy"], "critic": ["policy"]}
    agent_cfg.clip_actions = 1.0
    agent_cfg.num_steps_per_env = 32

    with launch_simulation(cfg, {"device": "cuda:0"}):
        raw_env = LearningBalanceEnv(cfg)
        wrapped = RslRlVecEnvWrapper(raw_env, clip_actions=1.0)
        try:
            runner = OnPolicyRunner(wrapped, to_dict(agent_cfg),
                                     log_dir=str(output), device="cuda:0")
            if args.checkpoint:
                runner.load(args.checkpoint, map_location="cuda:0")
            else:
                runner.learn(num_learning_iterations=args.iterations,
                             init_at_random_ep_len=False)
                runner.save(str(output / "final.pt"))

            policy = runner.get_inference_policy(device=raw_env.device)
            wrapped.reset()  # begin evaluation from fresh task states
            obs = wrapped.get_observations()
            returns = torch.zeros(raw_env.num_envs, device=raw_env.device)
            completed = []
            with torch.inference_mode():
                for _ in range(2 * raw_env.max_episode_length):
                    obs, rewards, dones, extras = wrapped.step(policy(obs))
                    returns += rewards
                    ended = dones.bool()
                    if ended.any():
                        completed.extend(returns[ended].cpu().tolist())
                        returns[ended] = 0.0
                    policy.reset(dones)
            if completed:
                print("Completed episodes:", len(completed))
                print("Mean evaluation return:", sum(completed) / len(completed))
            runner.export_policy_to_jit(path=str(output / "exported"), filename="policy.pt")
        finally:
            wrapped.close()

if __name__ == "__main__":
    main()
```

This evaluation is a quick batch rollout, not a rigorous fixed-number-of-episodes benchmark: shorter episodes contribute more completed episodes in a fixed duration. For comparisons, evaluate a fixed seed/scenario set and equal episode counts per condition.

```powershell
& 'C:\Users\jiaxi\IsaacLab\.venv\Scripts\python.exe' isaac_train_evaluate.py --num-envs 64 --iterations 100
& 'C:\Users\jiaxi\IsaacLab\.venv\Scripts\python.exe' isaac_train_evaluate.py --num-envs 8 --checkpoint '.\isaac_balance_run\final.pt'
```

### 12.8 `export_policy_to_jit` and `export_policy_to_onnx`

These runner methods export the inference model. They do not export your physics scene, reward, reset logic, or hardware transport. A `.pt` file can mean either a training checkpoint or a TorchScript model; file extension alone is insufficient to distinguish them.

```python
runner.export_policy_to_jit(path="exported", filename="policy.pt")
runner.export_policy_to_onnx(path="exported", filename="policy.onnx")
```

For the local MLP exporter, normalization in the actor is included. Observation assembly, joint ordering, clipping outside the actor, and actuator scaling still need to match. Recurrent/CNN exports can have different interfaces; inspect their input/output names and shapes.

### 12.9 `torch.jit.load`, `.eval()`, and `torch.inference_mode()`

`torch.jit.load` loads a TorchScript export, not an arbitrary RSL-RL training checkpoint. `.eval()` sets inference behavior; `inference_mode()` disables gradient bookkeeping. A plain MLP export takes a batched observation tensor.

```python
policy = torch.jit.load("exported/policy.pt", map_location="cpu").eval()
observation = torch.zeros((1, 4), dtype=torch.float32)  # Cartpole example shape
with torch.inference_mode():
    raw_action = policy(observation)
action = raw_action.clamp(-1.0, 1.0)
```

For deployment, `observation` must contain live measurements assembled in the training order. Zeros here only demonstrate the API shape.

## 13. Deployment and matching the training contract

Deployment means using a trained policy to compute actions from new measurements. It can mean playback in the same simulator, transfer to another simulator, or control on a physical robot. Only the first is automatically covered by a training-library playback script.

### 13.1 Save the policy's interface alongside the weights

The policy only sees numbers. Their ordering, reference frames, units, scaling, filtering, and history are part of the learned function. Keep a small manifest with the checkpoint. This example corresponds to the proposed 15-observation car task, not to the four-observation Cartpole example.

```python
import json
from pathlib import Path
manifest = {
    "observation_order": ["command_vx", "command_yaw_rate",
                          "vx_b", "vy_b", "vz_b", "wx_b", "wy_b", "wz_b",
                          "gravity_x_b", "gravity_y_b", "gravity_z_b",
                          "wheel_FL", "wheel_RL", "wheel_FR", "wheel_RR"],
    "observation_units": "m/s, rad/s, unit gravity direction, wheel rad/s",
    "action_order": ["left_side", "right_side"],
    "action_clip": [-1.0, 1.0],
    "action_scale_rad_per_s": 8.0,
    "policy_period_s": 0.02,
    "quaternion_order": "xyzw",
    "normalization": "record whether embedded in exported policy or external",
}
Path("policy_contract.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
```

Also retain joint signs/zero offsets, motor-to-joint gear ratios, gains/limits, initial pose, model revision, software versions, observation clipping, latency assumptions, and recurrent/history reset rules. An exported network does not infer these from the hardware.

### 13.2 SB3 export: `torch.nn.Module` and `torch.onnx.export`

`torch.nn.Module` wraps the deterministic policy action path. `register_buffer` stores non-trainable bounds with the module. `forward` defines its inference calculation. `torch.onnx.export` serializes a graph usable by ONNX Runtime. SB3's raw policy forward call does not automatically reproduce every postprocessing step of `predict`; this example explicitly clips actions.

This fragment is for a feedforward **continuous Box-action PPO MLP**, such as `MotorTargetEnv`, with no external `VecNormalize`. It is not a general exporter for discrete, recurrent, squashed, image, or dictionary policies.

```python
# export-example: sb3_box_onnx
import torch
from stable_baselines3 import PPO

class DeterministicBoxActor(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.policy = model.policy
        self.register_buffer("low", torch.as_tensor(model.action_space.low))
        self.register_buffer("high", torch.as_tensor(model.action_space.high))

    def forward(self, observation):
        action, _, _ = self.policy(observation, deterministic=True)
        return torch.maximum(self.low, torch.minimum(self.high, action))

model = PPO.load("motor_run/policy.zip", device="cpu")
model.policy.set_training_mode(False)
actor = DeterministicBoxActor(model).eval()
example_obs = torch.zeros((1, *model.observation_space.shape), dtype=torch.float32)
torch.onnx.export(
    actor, example_obs, "motor_policy.onnx",
    input_names=["observation"], output_names=["action"],
    opset_version=17, dynamo=False,
)
```

`dynamo=False` selects the legacy exporter explicitly for this small example; modern PyTorch also has a Dynamo-based exporter. ONNX export/runtime need their own compatible packages. If training used `VecNormalize`, supply the same frozen observation transform before this actor, or incorporate it into the graph and verify parity. [SB3 export guide](https://stable-baselines3.readthedocs.io/en/master/guide/export.html)

### 13.3 ONNX Runtime: `InferenceSession`, `get_inputs`, `get_outputs`, `run`

`InferenceSession` loads the model and chooses an execution provider. `get_inputs`/`get_outputs` reveal names, dtypes, and shapes. `run` evaluates requested outputs from a name-to-array input dictionary. Do not assume every exported model uses the same input name.

```python
import onnxruntime as ort
session = ort.InferenceSession("motor_policy.onnx", providers=["CPUExecutionProvider"])
for value in session.get_inputs():
    print(value.name, value.shape, value.type)
for value in session.get_outputs():
    print(value.name, value.shape, value.type)

obs = np.array([[0.0, 1.0, 0.0, 0.5]], dtype=np.float32)
input_name = session.get_inputs()[0].name
action = session.run(None, {input_name: obs})[0]
```

This is the one-input MLP case. A recurrent policy may require hidden-state inputs/outputs. ONNX Runtime also has a C++ API for integration into a C++ control program; that program still needs exactly the same observation builder and actuator map. [ONNX Runtime Python API](https://onnxruntime.ai/docs/api/python/api_summary.html)

### 13.4 Verify export parity before connecting a controller

Run the same real observation batch through the original and exported policy and compare actions. Include near-limit and representative recorded observations, not just zeros. Here `model` is the original SB3 model and `session` is its matching ONNX export.

```python
obs, info = MotorTargetEnv().reset(seed=123)
expected, _ = model.predict(obs, deterministic=True)
actual = session.run(None, {input_name: obs[None].astype(np.float32)})[0][0]
np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-5)
```

For RSL-RL, compare `runner.get_inference_policy()` on the matching observation group with the exported model's corresponding flat tensor input. Include action clipping and normalizers exactly once on each path.

### 13.5 Example observation assembly for a car

This pure function matches the 15-value contract in section 10. Its arguments must already be measured/estimated in the specified coordinates. Do not silently replace simulator truth with a differently defined sensor value.

```python
def build_car_observation(command, velocity_b, angular_velocity_b,
                          gravity_b, wheel_rates):
    obs = np.concatenate([
        np.asarray(command).reshape(2),
        np.asarray(velocity_b).reshape(3),
        np.asarray(angular_velocity_b).reshape(3),
        np.asarray(gravity_b).reshape(3),
        np.asarray(wheel_rates).reshape(4),
    ]).astype(np.float32)
    if obs.shape != (15,) or not np.isfinite(obs).all():
        raise ValueError("Invalid policy observation")
    return obs

obs = build_car_observation(
    [0.5, 0.0], [0.2, 0.0, 0.0], [0.0, 0.0, 0.02],
    [0.0, 0.0, -1.0], [1.3, 1.3, 1.4, 1.4],
)
```

Real chassis velocity usually comes from an estimator combining wheel odometry, IMU, and possibly vision. It is not directly available just because the simulator provides `root_link_lin_vel_b`. If you cannot measure a training observation reliably, redesign the observation or supply a validated estimator.

### 13.6 Action decoding for wheels or a quadruped

A wheel-speed policy output must become wheel-speed commands; it must not be sent directly as amperes or torque. This function returns a command only; it does not communicate with hardware.

```python
def decode_car_action(action):
    a = np.asarray(action, dtype=np.float32).reshape(2)
    if not np.isfinite(a).all():
        raise ValueError("Non-finite action")
    sides = 8.0 * np.clip(a, -1.0, 1.0)
    return sides[[0, 0, 1, 1]]  # [FL, RL, FR, RR] wheel rad/s
```

For a position-offset quadruped policy, the decoder instead creates joint targets, then the low-level controller computes torque. The policy can run at 50 Hz while the joint controller runs faster; the training simulation must represent that timing and actuation behavior.

```python
q_target = q_default + 0.25 * np.clip(action12, -1.0, 1.0)
q_target = np.clip(q_target, q_min, q_max)
tau = kp * (q_target - measured_q) + kd * (dq_target - measured_dq) + tau_ff
tau = np.clip(tau, -tau_limit, tau_limit)
```

These are joint-coordinate quantities. Apply motor sign, gearing, and zero-offset conversions at the motor interface, with the inverse conversions on measurements. Do not command the base's free-joint coordinates as if they were motors.

### 13.7 Timing and the hardware adapter boundary

There is no universal `IsaacLab.deploy_to_robot()` function that knows your CAN IDs, firmware, encoder units, state estimator, or emergency-stop behavior. Those belong to the robot interface.

The following is **pseudocode**, not a runnable hardware program. `read_sensors`, `estimate_state`, `send_motor_commands`, and `safe_stop` are functions you must implement for the actual controller. A Windows Python sleep loop is not a hard real-time motor-control guarantee.

```text
initialize policy, state estimator, motor mappings, limits, watchdog
set policy deadline using a monotonic clock
loop while enabled:
    read timestamped sensors
    if sensors stale, state invalid, or controller fault:
        safe_stop and leave loop
    estimate state in the training coordinate frames
    construct and normalize observation exactly as during training
    evaluate policy without gradients
    validate and decode bounded action
    update low-level controller target
    send motor commands and feed watchdog
    check deadline; on a missed-deadline fault, use defined fallback
    wait until next policy deadline
on exit: safe_stop
```

For deployment validation, proceed from offline recorded observations, to replay in the training simulator, to another simulator if useful, then to constrained hardware tests with appropriate motor limits and an independent stop path. Export success demonstrates numerical packaging, not successful physical control.

### 13.8 What changes when evaluating or deploying?

| Stage | Physics/sensors | Policy weights | Main operation |
|---|---|---|---|
| Training | Simulation with designed exploration/randomization | Updated | `model.learn(...)` or `runner.learn(...)` |
| Simulation evaluation | Simulation under chosen test conditions | Frozen | `predict(...)` or inference policy |
| Export verification | Recorded/generated observation inputs | Frozen | Compare original and exported outputs |
| Physical deployment | Actual sensors and state estimation | Usually frozen | Inference + hardware controller |

Keep the critic and optimizer for training/resume checkpoints. They are usually unnecessary in the deployed feedforward actor. Keep preprocessing, normalization, recurrent/history state where used, and output transformations because they affect the policy's behavior.

## 14. Debugging reference

| Symptom | Likely check | Minimal diagnostic/example |
|---|---|---|
| Cannot import a package | Wrong interpreter or missing dependency | `print(sys.executable)` |
| `mujoco.step` missing | Native API name | `mujoco.mj_step(model, data)` |
| Control array changes but robot does not | Array never written to simulator | `data.ctrl[:] = commands` |
| Wheel target uses address as angle | Address/value confusion | `angle = data.qpos[q_index]` |
| Wheel stops after rotating | Position drive hits target/range | Inspect `model.actuator_ctrlrange` |
| Reset observation has wrong shape | Space declaration disagrees with returned vector | `print(obs.shape, env.observation_space.shape)` |
| Array broadcasting error `(5,)` versus `(30,)` | `+` adds arrays instead of concatenating | `obs = np.concatenate([state5, history30]).astype(np.float32)` |
| Reset must return tuple | Single-env Gymnasium contract | `return obs, {}` |
| Too many/few values from step | Mixed raw-env and wrapper APIs | Use section 7.8's matching signature |
| `step()` never returns | Infinite viewer/game loop inside a transition | Advance a fixed number of substeps, then return |
| Stored history changes unexpectedly | Simulator array was saved by reference | `history.append(data.qpos.copy())` |
| Root orientation is nonsense | Quaternion ordering mismatch | Convert `wxyz` ↔ `xyzw` explicitly |
| Robots overlap after reset | Missing environment-origin offset | `pose[:, :3] += scene.env_origins[env_ids]` |
| High reward but bad behavior | Reward loophole or mismatched metric | Log actual speed error and falls |
| GPU slower than CPU | Small batch, copies, rendering, compilation | Benchmark after warm-up without viewer |
| Physics feels slow while watching | Wall clock slower than simulated clock | Compare simulated seconds / wall seconds |
| Reloaded policy behaves differently | Normalizer, joint order, action scale, or config mismatch | Compare policy outputs on identical observations |
| Training restarts from scratch | Created new model without loading | `model = PPO.load(path, env=env)` |
| TensorBoard is empty | Wrong directory or no event files | Point `--logdir` at the training output directory |
| Old Isaac tutorial does not run | Different API generation | Compare section 1 snapshot and installed source |

### 14.1 Minimal physical checks before spending a training budget

1. Reset repeatedly: states must be finite, legal, and above the floor as intended.
2. Send zero action: verify its actual meaning (zero torque, zero speed, or default pose).
3. Command each actuator alone at low magnitude: record sign, units, and response.
4. Validate one observation by independent calculation, such as Jacobian velocity or finite-difference position over simulation time.
5. Confirm that task failure and timeout return the correct flags.
6. Run random actions in a small batch, checking shapes/devices and logs.
7. Train briefly and verify save/load/evaluation before starting a long run.

## 15. Validation and source references

### 15.1 What was checked for this document

Checks performed on 2026-10-20:

| Check | Result and scope |
|---|---|
| Installed versions | Read from both environments; Isaac Lab source revision recorded in section 1 |
| Python examples | All 92 Python fenced blocks compiled for syntax; fragments were not all executed |
| Markdown | Parsed with a Markdown parser supporting tables; code fences balanced and local source-link targets checked |
| Custom MuJoCo environment | Passed SB3 `check_env`, including observation/action compatibility; identical seeded resets matched |
| PPO lifecycle | Ran 256 training transitions with the example environment, saved/reloaded a checkpoint, checked identical deterministic predictions, and evaluated two episodes |
| MuJoCo measurement/state APIs | Exercised state snapshot/restore and compared COM velocity from `mj_objectVelocity` with the COM Jacobian result |
| SB3 export | Exported the smoke-test actor to ONNX; ONNX Runtime actions matched SB3 predictions on 10 seeded reset observations within `rtol=1e-4`, `atol=1e-5` |
| Isaac Lab imports/configuration | Imported the guide's core classes, constructed custom task/actuator/converter configurations, and validated the RSL-RL configuration |
| Isaac Lab CLI | Queried installed CLI and RSL-RL training help to verify command structure |
| Isaac Lab runtime | Attempted four environments and one PPO iteration; stopped during scene creation because NVIDIA's configured floor USD could not be retrieved. No Isaac Lab training, playback, or export success is claimed |
| Standalone MJWarp | Core function signatures inspected; GPU stepping/graph snippets were not executed as part of this guide's checks |
| Viewer/camera/hardware | Not visually or physically tested during this document task |

The unavailable asset was:

```text
https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/6.2/Isaac/IsaacLab/Environments/Grid/default_ground_plane_checker_v1/default_ground_plane.usda
```

This result does not distinguish restricted connectivity from an unavailable asset revision. Confirm asset access and the checkout's asset version before interpreting it as an environment-code failure. The small MuJoCo training run verified the API lifecycle; it did not establish a good trained controller. No robot firmware, motor commands, or existing project simulation files were changed.

### 15.2 Official references

The examples and explanations here are original teaching examples informed by the installed source and these primary references. Links to `stable`/`develop` can change; the local snapshot in section 1 records what the examples target.

- [MuJoCo Python API and viewer](https://mujoco.readthedocs.io/en/stable/python.html)
- [MuJoCo function reference](https://mujoco.readthedocs.io/en/stable/APIreference/APIfunctions.html)
- [MuJoCo model/data types](https://mujoco.readthedocs.io/en/stable/APIreference/APItypes.html)
- [MuJoCo MJCF reference](https://mujoco.readthedocs.io/en/stable/XMLreference.html)
- [MuJoCo Warp API](https://mujoco.readthedocs.io/en/stable/mjwarp/api.html)
- [Gymnasium custom environment tutorial](https://gymnasium.farama.org/introduction/create_custom_env/)
- [Stable-Baselines3 PPO](https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html)
- [Stable-Baselines3 vector environments](https://stable-baselines3.readthedocs.io/en/master/guide/vec_envs.html)
- [Stable-Baselines3 callbacks](https://stable-baselines3.readthedocs.io/en/master/guide/callbacks.html)
- [Stable-Baselines3 model export](https://stable-baselines3.readthedocs.io/en/master/guide/export.html)
- [Isaac Lab migration to 3.0](https://isaac-sim.github.io/IsaacLab/develop/source/isaaclab/migrating.html)
- [Isaac Lab direct environment tutorial](https://isaac-sim.github.io/IsaacLab/develop/source/how-to/create_direct_rl_env.html)
- [Isaac Lab environment API](https://isaac-sim.github.io/IsaacLab/develop/source/api/lab/isaaclab.envs.html)
- [Isaac Lab actuator API](https://isaac-sim.github.io/IsaacLab/develop/source/api/lab/isaaclab.actuators.html)
- [Isaac Lab Newton backend](https://isaac-sim.github.io/IsaacLab/develop/source/concepts/physics-backends/newton.html)
- [Preparing an asset for Newton/MJWarp](https://isaac-sim.github.io/IsaacLab/develop/source/how-to/prepare_asset_for_newton.html)
- [NVIDIA course: Isaac Lab and Newton](https://docs.nvidia.com/learning/physical-ai/getting-started-with-newton/latest/index.html)
- [ONNX Runtime Python inference API](https://onnxruntime.ai/docs/api/python/api_summary.html)

### 15.3 Local source files used to resolve version differences

These paths are references to your installed checkout; the Markdown does not modify them.

- [Cartpole direct implementation](C:/Users/jiaxi/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/core/cartpole/cartpole_direct_env.py)
- [Cartpole direct configuration](C:/Users/jiaxi/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/core/cartpole/cartpole_direct_env_cfg.py)
- [Cartpole manager configuration](C:/Users/jiaxi/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/core/cartpole/cartpole_manager_env_cfg.py)
- [Cartpole PPO configuration](C:/Users/jiaxi/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/core/cartpole/agents/rsl_rl_ppo_cfg.py)
- [Articulation methods](C:/Users/jiaxi/IsaacLab/source/isaaclab/isaaclab/assets/articulation/base_articulation.py)
- [Articulation data members](C:/Users/jiaxi/IsaacLab/source/isaaclab/isaaclab/assets/articulation/base_articulation_data.py)
- [Actuator command interface](C:/Users/jiaxi/IsaacLab/source/isaaclab/isaaclab/actuators/actuator_collection.py)
- [Actuator configuration](C:/Users/jiaxi/IsaacLab/source/isaaclab/isaaclab/actuators/actuator_base_cfg.py)
- [MJCF converter configuration](C:/Users/jiaxi/IsaacLab/source/isaaclab/isaaclab/sim/converters/mjcf_converter_cfg.py)
- [Simulation launcher](C:/Users/jiaxi/IsaacLab/source/isaaclab/isaaclab/app/sim_launcher.py)
- [RSL-RL wrapper](C:/Users/jiaxi/IsaacLab/source/isaaclab_rl/isaaclab_rl/rsl_rl/vecenv_wrapper.py)
- [RSL-RL training entrypoint](C:/Users/jiaxi/IsaacLab/source/isaaclab_rl/isaaclab_rl/entrypoints/backends/train_rsl_rl.py)
- [RSL-RL playback/export usage](C:/Users/jiaxi/IsaacLab/source/isaaclab_rl/isaaclab_rl/entrypoints/backends/play_rsl_rl.py)
- [Installed RSL-RL runner](C:/Users/jiaxi/IsaacLab/.venv/Lib/site-packages/rsl_rl/runners/on_policy_runner.py)





