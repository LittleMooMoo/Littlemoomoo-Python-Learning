import berkeley_humanoid_lite_lowlevel.recoil as recoil

motor_id = 1
can_channel = "can1"
timeout_threshold = 0.1


def ping_motor(motor_id: int, can_channel: str) -> bool:
    bus = recoil.Bus(channel=can_channel, bitrate=1000000)
    try:
        return bus.ping(motor_id, timeout=timeout_threshold)
    finally:
        bus.stop()


def main() -> None:
    isonline = ping_motor(motor_id, can_channel)
    if isonline:
        print(f"{motor_id} is online at {can_channel}")
    elif not isonline:
        print(f"{motor_id} is offline at {can_channel}")
        raise RuntimeError(
            f"no response from {motor_id} within {timeout_threshold}")


if __name__ == "__main__":
    main()
