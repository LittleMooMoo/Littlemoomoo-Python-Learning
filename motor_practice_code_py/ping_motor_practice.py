import argparse
import berkeley_humanoid_lite_lowlevel.recoil as recoil


def get_args_ping():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--channel", type=str, default="can1")
    parser.add_argument("-i", "--motor_id", type=int, default=1)
    return parser.parse_args()


timeout_threshold = 0.1


def ping_motor(bus, motor_id) -> bool:
    return bus.ping(motor_id, timeout=timeout_threshold)


def main() -> None:
    args = get_args_ping()
    motor_id = args.motor_id
    can_channel = args.channel
    bus = recoil.Bus(channel=can_channel, bitrate=1000000)
    try:
        is_online = ping_motor(bus, motor_id)
        if is_online:
            print(f"{motor_id} is online at {can_channel}")
        else:
            print(f"{motor_id} is offline at {can_channel}")
            raise RuntimeError(
                f"no response from {motor_id} within {timeout_threshold}")
    finally:
        bus.stop()


if __name__ == "__main__":
    main()
