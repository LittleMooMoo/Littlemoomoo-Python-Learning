# Recoil Firmware Function Summary

This note summarizes the Python firmware wrapper used by the Berkeley Humanoid
Lite Recoil motor controllers.

Main import:

```python
import berkeley_humanoid_lite_lowlevel.recoil as recoil
```

In the quadruped reference repo, these functions are defined in:

```text
Control_Code/recoil/core.py
```

## Big Picture

The Python `recoil.Bus` object talks to motor firmware over CAN.

Your Python code calls a normal Python function:

```python
bus.ping(3, timeout=0.1)
```

The library turns that into a CAN frame, sends it on the CAN bus, waits for a
response frame, decodes the response bytes, and returns a Python value such as
`True`, `False`, `float`, `int`, `tuple`, or `None`.

## What `bus.ping()` Does

Definition from the Recoil wrapper:

```python
def ping(self, device_id: int, timeout=0.1) -> bool:
    self.transmit(CANFrame(device_id, Function.RECEIVE_PDO_1, size=1, data=b"\xCA"))
    rx_frame = self.receive(filter_device_id=device_id, filter_function=Function.TRANSMIT_PDO_1, timeout=timeout)
    if not rx_frame:
        return False
    rx_data = self.unpack("<BBBBBBBB", rx_frame.data)[0]
    return rx_data == 0xCA
```

Step by step:

1. You give it a motor CAN ID:

```python
device_id = 3
```

2. It sends a CAN frame to that motor:

```python
CANFrame(
    device_id=3,
    func_id=Function.RECEIVE_PDO_1,
    size=1,
    data=b"\xCA",
)
```

3. The data payload is one byte:

```python
b"\xCA"
```

That is hexadecimal `CA`, used like a tiny "are you there?" message.

4. It waits for a response from the same device ID and the expected response
function:

```python
self.receive(
    filter_device_id=device_id,
    filter_function=Function.TRANSMIT_PDO_1,
    timeout=timeout,
)
```

5. If no response arrives before timeout, `receive()` returns `None`, and
`ping()` returns:

```python
False
```

6. If a response arrives, it unpacks the response bytes and checks whether the
first byte is also `0xCA`.

7. Final return type:

```python
bool
```

So:

```python
True
```

means online.

```python
False
```

means no valid response before timeout.

## Data Types In And Out

For `ping()`:

```python
bus.ping(device_id: int, timeout: float) -> bool
```

Example:

```python
is_online = bus.ping(3, timeout=0.1)
```

Input:

```text
device_id: int
timeout: float seconds
```

Output:

```text
True or False
```

## Basic CAN Data Objects

### `DataFrame`

Definition shape:

```python
class DataFrame:
    def __init__(
        self,
        device_id: int = 0,
        func_id: int | None = None,
        size: int = 0,
        data: bytes | bytearray = b""
    ):
```

Stores:

```text
device_id: motor CAN ID
func_id: message function/category
size: number of bytes in data
data: raw bytes being sent or received
```

### `CANFrame`

Definition shape:

```python
class CANFrame(DataFrame):
    def __init__(
        self,
        device_id: int = 0,
        func_id: int | None = None,
        size: int = 0,
        data: bytes | bytearray = b""
    ):
```

`CANFrame` is the CAN-specific version of `DataFrame`.

Important rule:

```python
assert self.size <= 8
```

Classic CAN frames carry up to 8 data bytes.

## Function IDs

The wrapper has a `Function` class. These are message categories.

Common ones:

```python
Function.NMT
Function.RECEIVE_PDO_1
Function.TRANSMIT_PDO_1
Function.RECEIVE_PDO_2
Function.TRANSMIT_PDO_2
Function.RECEIVE_SDO
Function.TRANSMIT_SDO
Function.FLASH
Function.HEARTBEAT
```

Simple meaning:

```text
NMT: mode control
PDO: fast runtime data
SDO: slower parameter read/write
FLASH: store/load settings
HEARTBEAT: watchdog feed
```

## Modes

The wrapper has a `Mode` class.

Most important modes for you:

```python
recoil.Mode.IDLE
recoil.Mode.DAMPING
recoil.Mode.POSITION
```

Meaning:

```text
IDLE: motor controller is not actively holding position
DAMPING: passive damping behavior
POSITION: closed-loop position control
```

## Bus Setup And Shutdown

### Open a CAN bus

Definition:

```python
def __init__(self, channel: str, bitrate: int = 1000000):
```

Example:

```python
bus = recoil.Bus(channel="can1", bitrate=1000000)
```

Input:

```text
channel: string like "can1"
bitrate: integer, usually 1000000
```

Output:

```text
a Bus object
```

### Stop the bus

Definition:

```python
def stop(self):
```

Example:

```python
bus.stop()
```

Use this in `finally` so the CAN interface is closed cleanly.

## Health And Mode Functions

### `ping`

```python
def ping(self, device_id: int, timeout=0.1) -> bool:
```

Checks whether one motor responds.

### `feed`

```python
def feed(self, device_id: int) -> None:
```

Sends watchdog heartbeat.

Use this before and during active control:

```python
bus.feed(device_id)
```

### `set_mode`

```python
def set_mode(self, device_id: int, mode: Mode) -> None:
```

Example:

```python
bus.set_mode(device_id, recoil.Mode.POSITION)
bus.set_mode(device_id, recoil.Mode.IDLE)
```

It sends a mode command using `Function.NMT`.

## Parameter Read/Write Pattern

The wrapper has private helper functions:

```python
_read_parameter_f32(...) -> float | None
_read_parameter_i32(...) -> int | None
_read_parameter_u32(...) -> int | None
_write_parameter_f32(...) -> None
_write_parameter_i32(...) -> None
_write_parameter_u32(...) -> None
```

Most public functions are small wrappers around those helpers.

Example:

```python
def read_position_kp(self, device_id: int) -> float | None:
    return self._read_parameter_f32(device_id, Parameter.POSITION_CONTROLLER_POSITION_KP)
```

Example:

```python
def write_position_kp(self, device_id: int, value: float):
    self._write_parameter_f32(device_id, Parameter.POSITION_CONTROLLER_POSITION_KP, value)
```

That means:

```python
bus.write_position_kp(3, 0.25)
```

is really:

```text
write float 0.25 to firmware parameter POSITION_CONTROLLER_POSITION_KP
for motor ID 3
```

## Position Controller Functions

These are the most useful for early motor control.

### Gains

```python
read_position_kp(device_id: int) -> float | None
write_position_kp(device_id: int, value: float)

read_position_kd(device_id: int) -> float | None
write_position_kd(device_id: int, value: float)

read_position_ki(device_id: int) -> float | None
write_position_ki(device_id: int, value: float)
```

Notes:

```python
write_position_kd(...)
```

internally writes velocity `kp`:

```python
def write_position_kd(self, device_id: int, value: float):
    self.write_velocity_kp(device_id, value)
```

### Limits

```python
read_torque_limit(device_id: int) -> float | None
write_torque_limit(device_id: int, value: float)

read_velocity_limit(device_id: int) -> float | None
write_velocity_limit(device_id: int, value: float)

read_position_limit_lower(device_id: int) -> float | None
write_position_limit_lower(device_id: int, value: float)

read_position_limit_upper(device_id: int) -> float | None
write_position_limit_upper(device_id: int, value: float)
```

### Target And Feedback

```python
read_position_target(device_id: int) -> float | None
write_position_target(device_id: int, value: float)

read_position_measured(device_id: int) -> float | None

read_velocity_target(device_id: int) -> float | None
write_velocity_target(device_id: int, value: float)

read_velocity_measured(device_id: int) -> float | None

read_torque_target(device_id: int) -> float | None
write_torque_target(device_id: int, value: float)

read_torque_measured(device_id: int) -> float | None
```

## Fast PDO Motion Functions

These are used inside control loops.

### `transmit_pdo_2`

Definition:

```python
def transmit_pdo_2(self, device_id: int, position_target: float, velocity_target: float):
    self.transmit(CANFrame(
        device_id,
        Function.RECEIVE_PDO_2,
        size=8,
        data=struct.pack("<ff", position_target, velocity_target)
    ))
```

Inputs:

```text
device_id: int
position_target: float
velocity_target: float
```

Output:

```text
None
```

Important data detail:

```python
struct.pack("<ff", position_target, velocity_target)
```

means pack two 32-bit floats into 8 bytes:

```text
4 bytes for position_target
4 bytes for velocity_target
```

Use it when you only want to send a command:

```python
bus.transmit_pdo_2(device_id, target_position, 0.0)
```

### `receive_pdo_2`

Definition shape:

```python
def receive_pdo_2(self, device_id: int) -> tuple:
```

It returns:

```python
(measured_position, measured_velocity)
```

If timeout/no response:

```python
(None, None)
```

### `write_read_pdo_2`

Definition:

```python
def write_read_pdo_2(self, device_id: int, position_target: float, velocity_target: float) -> tuple:
    self.transmit_pdo_2(device_id, position_target, velocity_target)
    return self.receive_pdo_2(device_id)
```

Meaning:

```text
send target position and velocity
then immediately wait for measured position and velocity
```

Example:

```python
measured_position, measured_velocity = bus.write_read_pdo_2(
    device_id,
    target_position,
    0.0,
)
```

## Flash Functions

```python
load_settings_from_flash(device_id: int) -> None
store_settings_to_flash(device_id: int) -> None
```

Use carefully.

Do not call `store_settings_to_flash()` inside a loop. Flash memory has limited
write life.

## Motor/Encoder Configuration Functions

These are more advanced. They change firmware-level actuator configuration.

```python
read_gear_ratio(device_id: int) -> float | None
write_gear_ratio(device_id: int, value: float)

read_motor_pole_pairs(device_id: int) -> int | None
write_motor_pole_pairs(device_id: int, value: int)

read_motor_torque_constant(device_id: int) -> float | None
write_motor_torque_constant(device_id: int, value: float)

read_motor_phase_order(device_id: int) -> int | None
write_motor_phase_order(device_id: int, value: int)

read_motor_calibration_current(device_id: int) -> float | None
write_motor_calibration_current(device_id: int, value: float)

read_encoder_cpr(device_id: int) -> int | None
write_encoder_cpr(device_id: int, value: int)

read_encoder_position_offset(device_id: int) -> float | None
write_encoder_position_offset(device_id: int, value: float)

read_encoder_velocity_filter_alpha(device_id: int) -> float | None
write_encoder_velocity_filter_alpha(device_id: int, value: float)

read_encoder_flux_offset(device_id: int) -> float | None
write_encoder_flux_offset(device_id: int, value: float)
```

For normal walking tests, avoid changing these unless you are intentionally
configuring or calibrating the actuator.

## Bandwidth Helper Functions

These compute filter/gain values and write them to firmware.

```python
set_current_bandwidth(device_id, bandwidth_hz, phase_resistance, phase_inductance)
set_torque_bandwidth(device_id, bandwidth_hz, position_loop_rate=2000)
set_bus_voltage_bandwidth(device_id, bandwidth_hz, bus_voltage_update_rate=10000)
set_encoder_velocity_bandwidth(device_id, bandwidth_hz, encoder_update_rate=10000)
```

These are advanced tuning helpers.

## First Safe Real-Motor Pattern

```python
import time
import berkeley_humanoid_lite_lowlevel.recoil as recoil


channel = "can1"
device_id = 3

bus = recoil.Bus(channel=channel, bitrate=1000000)

try:
    if not bus.ping(device_id, timeout=0.1):
        print("offline")
        raise RuntimeError(f"Motor {device_id} did not respond on {channel}")

    print("online")

    start_position = bus.read_position_measured(device_id)
    if start_position is None:
        raise RuntimeError("Could not read measured position")

    bus.write_position_kp(device_id, 0.003)
    bus.write_position_kd(device_id, 0.001)
    bus.write_torque_limit(device_id, 0.03)

    bus.set_mode(device_id, recoil.Mode.POSITION)
    bus.feed(device_id)

    for _ in range(100):
        bus.feed(device_id)
        bus.transmit_pdo_2(device_id, start_position, 0.0)
        time.sleep(0.01)

finally:
    bus.set_mode(device_id, recoil.Mode.IDLE)
    bus.stop()
```

This pattern does not intentionally move the motor. It reads where the motor
already is, then gently holds that position.
