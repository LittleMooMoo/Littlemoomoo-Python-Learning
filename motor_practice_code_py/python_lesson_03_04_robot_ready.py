"""
python_lesson_03_04_robot_ready.py

Lesson 03 + Lesson 04:
    - tuple: fixed xyz positions
    - dataclass: robot configuration data
    - class: one reusable motor controller object
    - Berkeley Humanoid Lite Recoil API: how Python talks to real motors

This file is safe to run on your normal computer because REAL_HARDWARE is False.
It uses a fake CAN bus first, so you can learn the control pattern before
touching the robot.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import time
from typing import Protocol


# ============================================================
# SAFETY SWITCH
# ============================================================

# Keep this False while learning.
# Only change it to True on the robot computer after:
#   1. CAN interface is up, such as can1/can2/can3/can4.
#   2. The robot is lifted or safely supported.
#   3. You know which motor ID you are commanding.
#   4. Your hand is ready to stop the program with Ctrl+C.
REAL_HARDWARE = False


# ============================================================
# LESSON 03: TUPLES FOR ROBOT POSITIONS
# ============================================================

# A tuple is good when the shape of the data should not change.
# A foot position is always x, y, z, so tuple[float, float, float] fits well.
FootPosition = tuple[float, float, float]

NEUTRAL_FOOT: FootPosition = (-20.47, 84.26, 353.30)
SLEEPY_FOOT: FootPosition = (0.0, 84.26, 410.0)


def print_foot_position(label: str, position: FootPosition) -> None:
    x, y, z = position
    print(f"{label}: x={x:.2f}, y={y:.2f}, z={z:.2f}")


def distance_between(a: FootPosition, b: FootPosition) -> float:
    ax, ay, az = a
    bx, by, bz = b
    return math.sqrt((ax - bx) ** 2 + (ay - by) ** 2 + (az - bz) ** 2)


# ============================================================
# LESSON 03: DATACLASSES FOR CONFIGURATION
# ============================================================

# A dataclass is excellent for robot configuration.
# It says, "this object stores named fields, and those fields matter."


@dataclass(frozen=True)
class JointGains:
    kp: float
    kd: float
    torque_limit: float


@dataclass(frozen=True)
class JointConfig:
    leg_name: str
    joint_name: str
    can_channel: str
    can_id: int
    motor_sign: float
    homing_angle_rad: float
    gains: JointGains


@dataclass
class MotorFeedback:
    measured_position: float
    measured_velocity: float


SAFE_STARTUP_GAINS = JointGains(kp=0.003, kd=0.001, torque_limit=0.03)
NORMAL_MOVE_GAINS = JointGains(kp=0.25, kd=0.015, torque_limit=1.30)

FRONT_RIGHT_HIP = JointConfig(
    leg_name="front_right",
    joint_name="hip",
    can_channel="can1",
    can_id=3,
    motor_sign=-1.0,
    homing_angle_rad=-0.465499025237436,
    gains=NORMAL_MOVE_GAINS,
)

FRONT_RIGHT_THIGH = JointConfig(
    leg_name="front_right",
    joint_name="thigh",
    can_channel="can1",
    can_id=2,
    motor_sign=1.0,
    homing_angle_rad=1.11367455104935,
    gains=NORMAL_MOVE_GAINS,
)

FRONT_RIGHT_SHANK = JointConfig(
    leg_name="front_right",
    joint_name="shank",
    can_channel="can1",
    can_id=1,
    motor_sign=1.0,
    homing_angle_rad=2.68814970801254,
    gains=NORMAL_MOVE_GAINS,
)


# ============================================================
# SMALL SAFETY HELPERS
# ============================================================

def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def smooth_step(start: float, end: float, phase: float) -> float:
    """Cosine interpolation with gentle start and stop."""
    phase = clamp(phase, 0.0, 1.0)
    s = 0.5 * (1.0 - math.cos(math.pi * phase))
    return start + (end - start) * s


def parse_float(text: str) -> float:
    try:
        return float(text)
    except ValueError as exc:
        raise ValueError("Please enter a number.") from exc


def joint_angle_delta_to_raw_motor_delta(
    joint_delta_rad: float,
    motor_sign: float,
    gear_ratio: float = 17.0,
) -> float:
    """
    Convert output-side joint angle change to raw motor position change.

    The quadruped control repo uses this idea:
        raw_delta = motor_sign * joint_delta_rad * gear_ratio
    """
    return motor_sign * joint_delta_rad * gear_ratio


# ============================================================
# LESSON 04: BUS INTERFACE
# ============================================================

# A Protocol says which methods our MotorController expects.
# The fake bus and the real Recoil bus both match this shape.


class PositionBus(Protocol):
    def ping(self, device_id: int, timeout: float = 0.1) -> bool:
        ...

    def write_position_kp(self, device_id: int, value: float) -> None:
        ...

    def write_position_kd(self, device_id: int, value: float) -> None:
        ...

    def write_torque_limit(self, device_id: int, value: float) -> None:
        ...

    def read_position_measured(self, device_id: int) -> float | None:
        ...

    def set_mode(self, device_id: int, mode: object) -> None:
        ...

    def feed(self, device_id: int) -> None:
        ...

    def transmit_pdo_2(
        self,
        device_id: int,
        position_target: float,
        velocity_target: float,
    ) -> None:
        ...

    def write_read_pdo_2(
        self,
        device_id: int,
        position_target: float,
        velocity_target: float,
    ) -> tuple[float | None, float | None]:
        ...

    def stop(self) -> None:
        ...


class FakeMode:
    IDLE = "IDLE"
    POSITION = "POSITION"
    DAMPING = "DAMPING"


class FakeRecoilBus:
    """
    A fake version of berkeley_humanoid_lite_lowlevel.recoil.Bus.

    This lets you practice the real function names:
        ping
        write_position_kp
        write_position_kd
        write_torque_limit
        set_mode
        feed
        transmit_pdo_2
        write_read_pdo_2
        read_position_measured
        stop
    """

    def __init__(self, channel: str, bitrate: int = 1000000):
        self.channel = channel
        self.bitrate = bitrate
        self.mode_by_id: dict[int, object] = {}
        self.position_by_id: dict[int, float] = {}
        self.velocity_by_id: dict[int, float] = {}
        print(f"FakeRecoilBus opened channel={channel}, bitrate={bitrate}")

    def ping(self, device_id: int, timeout: float = 0.1) -> bool:
        print(f"ping ID {device_id} on {self.channel}")
        return True

    def write_position_kp(self, device_id: int, value: float) -> None:
        print(f"write_position_kp ID {device_id}: {value}")

    def write_position_kd(self, device_id: int, value: float) -> None:
        print(f"write_position_kd ID {device_id}: {value}")

    def write_torque_limit(self, device_id: int, value: float) -> None:
        print(f"write_torque_limit ID {device_id}: {value}")

    def read_position_measured(self, device_id: int) -> float | None:
        return self.position_by_id.get(device_id, 0.0)

    def set_mode(self, device_id: int, mode: object) -> None:
        self.mode_by_id[device_id] = mode
        print(f"set_mode ID {device_id}: {mode}")

    def feed(self, device_id: int) -> None:
        print(f"feed watchdog ID {device_id}")

    def transmit_pdo_2(
        self,
        device_id: int,
        position_target: float,
        velocity_target: float,
    ) -> None:
        old_position = self.position_by_id.get(device_id, 0.0)
        self.position_by_id[device_id] = position_target
        self.velocity_by_id[device_id] = velocity_target
        print(
            f"transmit_pdo_2 ID {device_id}: "
            f"{old_position:+.3f} -> {position_target:+.3f}"
        )

    def write_read_pdo_2(
        self,
        device_id: int,
        position_target: float,
        velocity_target: float,
    ) -> tuple[float | None, float | None]:
        self.transmit_pdo_2(device_id, position_target, velocity_target)
        return (
            self.position_by_id[device_id],
            self.velocity_by_id[device_id],
        )

    def stop(self) -> None:
        print(f"FakeRecoilBus stopped channel={self.channel}")


def open_bus(channel: str) -> tuple[PositionBus, object]:
    """
    Return either a fake bus or the real Berkeley Recoil bus.

    Real hardware import used in the quadruped repo:
        import berkeley_humanoid_lite_lowlevel.recoil as recoil
    """
    if not REAL_HARDWARE:
        return FakeRecoilBus(channel=channel, bitrate=1000000), FakeMode

    import berkeley_humanoid_lite_lowlevel.recoil as recoil

    return recoil.Bus(channel=channel, bitrate=1000000), recoil.Mode


# ============================================================
# LESSON 04: CLASS FOR ONE REAL-STYLE MOTOR CONTROLLER
# ============================================================

class MotorController:
    """
    A small wrapper around the Recoil firmware API.

    The class stores config and remembers the most recent command.
    The bus object does the actual CAN communication.
    """

    def __init__(self, config: JointConfig, bus: PositionBus, mode: object):
        self.config = config
        self.bus = bus
        self.mode = mode
        self.active_target: float | None = None

    @property
    def can_id(self) -> int:
        return self.config.can_id

    @property
    def label(self) -> str:
        return f"{self.config.leg_name}_{self.config.joint_name}"

    def check_online(self) -> None:
        if not self.bus.ping(self.can_id):
            raise RuntimeError(f"{self.label} ID {self.can_id} did not respond.")

    def write_gains(self, gains: JointGains) -> None:
        self.bus.write_position_kp(self.can_id, gains.kp)
        self.bus.write_position_kd(self.can_id, gains.kd)
        self.bus.write_torque_limit(self.can_id, gains.torque_limit)

    def measured_position(self) -> float:
        position = self.bus.read_position_measured(self.can_id)
        if position is None:
            raise RuntimeError(f"Could not read {self.label} measured position.")
        return float(position)

    def enter_position_mode_at_current_position(self) -> None:
        start = self.measured_position()
        self.active_target = start

        self.write_gains(SAFE_STARTUP_GAINS)
        self.bus.set_mode(self.can_id, self.mode.POSITION)
        self.bus.feed(self.can_id)
        self.bus.transmit_pdo_2(self.can_id, start, 0.0)

        print(f"{self.label}: holding current position {start:+.3f}")

    def command_position(self, raw_target: float) -> MotorFeedback:
        self.bus.feed(self.can_id)
        measured_position, measured_velocity = self.bus.write_read_pdo_2(
            self.can_id,
            raw_target,
            0.0,
        )

        if measured_position is None or measured_velocity is None:
            raise RuntimeError(f"No PDO feedback from {self.label}.")

        self.active_target = raw_target
        return MotorFeedback(
            measured_position=float(measured_position),
            measured_velocity=float(measured_velocity),
        )

    def move_relative_smoothly(
        self,
        delta_raw: float,
        duration_s: float = 1.0,
        rate_hz: float = 20.0,
    ) -> None:
        """
        Move by a small raw motor-position delta using repeated PDO commands.

        On real hardware, use small delta_raw values first. A big target jump is
        not a good first robot test.
        """
        start = self.measured_position()
        target = start + delta_raw
        steps = max(2, int(duration_s * rate_hz))

        self.write_gains(self.config.gains)

        for step in range(steps + 1):
            phase = step / steps
            command = smooth_step(start, target, phase)
            feedback = self.command_position(command)

            print(
                f"{self.label}: cmd={command:+.3f}, "
                f"measured={feedback.measured_position:+.3f}, "
                f"vel={feedback.measured_velocity:+.3f}"
            )

            time.sleep(1.0 / rate_hz)

    def idle(self) -> None:
        self.write_gains(SAFE_STARTUP_GAINS)
        self.bus.set_mode(self.can_id, self.mode.IDLE)
        print(f"{self.label}: IDLE")


class LegController:
    """A class can also group three motors into one leg."""

    def __init__(self, name: str, motors: list[MotorController]):
        self.name = name
        self.motors = motors

    def enter_position_mode(self) -> None:
        for motor in self.motors:
            motor.enter_position_mode_at_current_position()

    def idle(self) -> None:
        for motor in self.motors:
            motor.idle()

    def print_summary(self) -> None:
        print(f"\n{self.name} motor summary")
        for motor in self.motors:
            cfg = motor.config
            print(
                f"  {cfg.joint_name:5s} ID={cfg.can_id:2d} "
                f"channel={cfg.can_channel} sign={cfg.motor_sign:+.1f}"
            )


# ============================================================
# RUNNABLE LESSON DEMO
# ============================================================

def run_tuple_and_dataclass_demo() -> None:
    print("\n=== tuple demo: foot positions ===")
    print_foot_position("neutral", NEUTRAL_FOOT)
    print_foot_position("sleepy", SLEEPY_FOOT)
    print(
        "distance neutral -> sleepy:",
        f"{distance_between(NEUTRAL_FOOT, SLEEPY_FOOT):.2f} mm",
    )

    print("\n=== dataclass demo: joint config ===")
    print(FRONT_RIGHT_HIP)

    joint_delta = 0.10
    raw_delta = joint_angle_delta_to_raw_motor_delta(
        joint_delta,
        FRONT_RIGHT_HIP.motor_sign,
    )
    print(
        f"{joint_delta:+.3f} rad output-side hip delta "
        f"becomes {raw_delta:+.3f} raw motor delta"
    )


def run_fake_recoil_motor_demo() -> None:
    print("\n=== fake Recoil motor-control demo ===")
    bus, mode = open_bus(FRONT_RIGHT_HIP.can_channel)

    hip = MotorController(FRONT_RIGHT_HIP, bus, mode)

    try:
        hip.check_online()
        hip.enter_position_mode_at_current_position()

        delta_text = input(
            "Enter a small raw relative move, or press Enter for 0.25: "
        )
        delta_raw = 0.25 if delta_text == "" else parse_float(delta_text)
        delta_raw = clamp(delta_raw, -1.0, 1.0)

        hip.move_relative_smoothly(
            delta_raw=delta_raw,
            duration_s=1.0,
            rate_hz=10.0,
        )

    finally:
        hip.idle()
        bus.stop()


def run_fake_leg_demo() -> None:
    print("\n=== fake leg class demo ===")
    bus, mode = open_bus(FRONT_RIGHT_HIP.can_channel)

    leg = LegController(
        "front_right",
        [
            MotorController(FRONT_RIGHT_HIP, bus, mode),
            MotorController(FRONT_RIGHT_THIGH, bus, mode),
            MotorController(FRONT_RIGHT_SHANK, bus, mode),
        ],
    )

    try:
        leg.print_summary()
        leg.enter_position_mode()
    finally:
        leg.idle()
        bus.stop()


def main() -> None:
    print("Lesson 03/04: Python data -> Recoil motor-control pattern")
    print("REAL_HARDWARE =", REAL_HARDWARE)

    run_tuple_and_dataclass_demo()
    run_fake_recoil_motor_demo()
    run_fake_leg_demo()

    print("\nKey real-hardware pattern:")
    print("1. bus = recoil.Bus(channel='can1', bitrate=1000000)")
    print("2. bus.ping(can_id)")
    print("3. bus.read_position_measured(can_id)")
    print("4. bus.write_position_kp/kd and bus.write_torque_limit")
    print("5. bus.set_mode(can_id, recoil.Mode.POSITION)")
    print("6. bus.feed(can_id)")
    print("7. bus.transmit_pdo_2(can_id, target_position, 0.0)")
    print("8. bus.set_mode(can_id, recoil.Mode.IDLE) in finally")

    print("\nPractice ideas:")
    print("- Add a JointConfig for front_left_hip on can2 with CAN ID 6.")
    print("- Add a method that reads current position and holds it for 2 seconds.")
    print("- Change run_fake_leg_demo so it checks ping for all three motors.")
    print("- Add a function that converts joint angle delta to raw target position.")
    print("- Build a tiny list of FootPosition tuples and print the distance between each one.")


if __name__ == "__main__":
    main()
