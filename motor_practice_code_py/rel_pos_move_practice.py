import time
import argparse
import math
import berkeley_humanoid_lite_lowlevel.recoil as recoil
from ping_motor_practice import ping_motor


def get_args_move():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--channel", type=str, default="can1")
    parser.add_argument("-i", "--motor_id", type=int, default=1)
    parser.add_argument("-d", "--delta_raw", type=float, default=0.1)
    return parser.parse_args()


timeout_threshold = 0.1

kp = 0.2
kd = 0.005
torque_limit = 0.4
move_time = 5.0


def ping_for_response(bus, motor_id):
    is_online = ping_motor(bus, motor_id)
    if is_online:
        print(f"{motor_id} is online")
    else:
        print(f"{motor_id} is offline")
        raise RuntimeError(
            f"no response from {motor_id} within {timeout_threshold}")


def read_current_raw_position(bus, motor_id):
    raw_position = bus.read_position_measured(motor_id)

    if raw_position is not None:
        return raw_position
    else:
        raise RuntimeError(f"no response from {motor_id} for position"
                           )


def motor_trajectory(bus, motor_id, delta_raw):
    raw_position = read_current_raw_position(bus, motor_id)
    target_position = raw_position + delta_raw

    start_time = time.perf_counter()

    while True:
        elapsed_time = time.perf_counter() - start_time
        phase = elapsed_time / move_time
        if phase > 1.0:
            phase = 1.0

        trajectory = 0.5 * (1.0 - math.cos(math.pi * phase))
        command_position = raw_position + (
            target_position - raw_position) * trajectory
        print(command_position)

        bus.feed(motor_id)
        bus.transmit_pdo_2(motor_id, command_position, 0.0)

        if phase >= 1.0:
            break
        time.sleep(0.01)
    return target_position


def main() -> None:
    args = get_args_move()
    motor_id = args.motor_id
    can_channel = args.channel
    delta_raw = args.delta_raw

    bus = recoil.Bus(channel=can_channel, bitrate=1000000)
    try:
        ping_for_response(bus, motor_id)
        read_current_raw_position(bus, motor_id)

        bus.write_position_kp(motor_id, kp)
        bus.write_position_kd(motor_id, kd)
        bus.write_torque_limit(motor_id, torque_limit)
        bus.set_mode(motor_id, recoil.Mode.POSITION)
        bus.feed(motor_id)
        time.sleep(0.1)

        motor_trajectory(bus, motor_id, delta_raw)
        while True:
            target_position = motor_trajectory(bus, motor_id, delta_raw)
            bus.feed(motor_id)
            bus.transmit_pdo_2(motor_id, target_position, 0.0)
            time.sleep(0.01)

    except KeyboardInterrupt:
        print("user stopped")
    finally:
        bus.set_mode(motor_id, recoil.Mode.IDLE)
        bus.stop()


if __name__ == "__main__":
    main()
