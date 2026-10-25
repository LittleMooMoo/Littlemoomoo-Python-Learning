import mujoco
import numpy as np
from pathlib import Path
import warp as wp
import mujoco_warp as mjw
import torch
import math
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
import isaaclab.sim as sim_utils
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab_newton.physics import NewtonCfg, MJWarpSolverCfg
from collections.abc import Sequence
from isaaclab.assets import ArticulationCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.sensors import ImuCfg

# Similar to the manager-based environments, a configuration class is defined
# for the task to hold settings for the simulation parameters, the scene, the actors, and the task.

CAR_XML_PATH = Path(__file__).resolve().parent/"diff_car.xml"


@configclass
class DiffCarSceneCfg(InteractiveSceneCfg):
    # diff_car.xml already contains a collision plane. Adding another ground
    # here creates two overlapping contact surfaces in the solver.
    ground = None
    car = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Car",
        spawn=sim_utils.MjcfFileCfg(
            asset_path=str(CAR_XML_PATH),
            fix_base=False,
        ),
        actuators={
            "wheels": ImplicitActuatorCfg(
                joint_names_expr=["left_axle", "right_axle"],
                # Velocity servo: the wheels can keep rolling instead of
                # being pulled back toward a fixed absolute angle.
                stiffness=0.0,
                damping=1.0,
                joint_effort_limit=8.0,  # N m per wheel; tested for swing-up
            ),
        },
    )
    imu = ImuCfg(
        prim_path="{ENV_REGEX_NS}/Car/Geometry/chasis",
        update_period=0.0,
    )


@configclass
class DiffCarEnvCfg(DirectRLEnvCfg):
    action_space = 2
    observation_space = 14
    state_space = 0
    task_stage = "swing_up_from_flat_then_balance"
    reset_pitch_degrees = 0.0
    reset_pitch_noise_degrees = 1.0
    max_tilt_degrees = 108.0  # flat is 90 deg from upright, so it must be allowed
    wheel_speed_limit = 30.0  # rad/s
    turn_action_scale = 0.25
    reward_discount = 0.995  # must match PPO gamma for potential-based shaping
    balance_hold_seconds = 2.0
    wheel_radius = 0.5
    wheel_axle_offset = 0.8  # chassis origin to axle, along local X
    decimation = 4
    episode_length_s = 20.0
    sim = sim_utils.SimulationCfg(
        dt=1 / 240,
        physics=NewtonCfg(solver_cfg=MJWarpSolverCfg(
            integrator="implicitfast")),
    )
    scene = DiffCarSceneCfg(
        num_envs=16,
        env_spacing=4.0,
    )


class DiffCarEnv(DirectRLEnv):
    cfg: DiffCarEnvCfg

    def __init__(self, cfg: DiffCarEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self.car = self.scene["car"]
        self.imu = self.scene["imu"]
        wheel_ids, _ = self.car.find_joints(
            ["left_axle", "right_axle"],
            preserve_order=True,
        )
        self.wheel_ids = torch.tensor(
            wheel_ids, dtype=torch.long, device=self.device
        )
        self.actions = torch.zeros(
            (self.num_envs, 2),
            dtype=torch.float32,
            device=self.device,
        )
        self.wheel_pos = torch.zeros(
            (self.num_envs, 2),
            dtype=torch.float32,
            device=self.device,
        )
        self.wheel_vel = torch.zeros(
            (self.num_envs, 2),
            dtype=torch.float32,
            device=self.device,
        )
        self.imu_gyro = torch.zeros(
            (self.num_envs, 3),
            dtype=torch.float32,
            device=self.device,
        )
        self.imu_lin_accel = torch.zeros(
            (self.num_envs, 3),
            dtype=torch.float32,
            device=self.device,
        )
        self.imu_quat = torch.zeros(
            (self.num_envs, 4),
            dtype=torch.float32,
            device=self.device,
        )
        self.imu_quat[:, 3] = 1.0
        self.chasis_pos = torch.zeros(
            (self.num_envs, 3),
            dtype=torch.float32,
            device=self.device,
        )
        # One scalar per environment: (N,), matching chasis_pos[:, 2].
        self.reference_chasis_pos = torch.full(
            (self.num_envs,), self.cfg.wheel_radius + self.cfg.wheel_axle_offset,
            device=self.device,
        )
        self.chasis_vel = torch.zeros((self.num_envs, 3), device=self.device)
        # Consecutive balanced steps, not just time survived.
        self.upright_step_count = torch.zeros(
            self.num_envs, device=self.device)
        self.previous_potential = torch.zeros(
            self.num_envs, device=self.device)

    def _get_observations(self) -> dict:
        self._read_measurements()

        obs = torch.cat(
            (
                self.wheel_pos,
                self.wheel_vel,
                self.imu_gyro,
                self.imu_lin_accel,
                self.imu_quat,
            ),
            dim=1,
        )
        bad = ~torch.isfinite(obs)

        if bad.any():
            indices = bad.nonzero(as_tuple=False)
            raise RuntimeError(
                f"Invalid [environment, feature] entries: {indices[:10].tolist()}"
            )
        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        self._read_measurements()
        total_reward = compute_rewards(self.chasis_pos,
                                       self.reference_chasis_pos,
                                       self.imu_quat,
                                       self.imu_gyro,
                                       self.chasis_vel,
                                       self.actions,
                                       self.cfg.wheel_axle_offset)
        tilt_error = compute_orientation_error(self.imu_quat)
        potential = 1.0 - 0.5 * tilt_error
        # The discounted potential difference credits lifting progress without
        # paying repeatedly for rocking up and down. Real failures end at zero
        # potential; time-limit truncations retain it for PPO bootstrapping.
        next_potential = torch.where(self.reset_terminated, 0.0, potential)
        total_reward += 2.0 * (
            self.cfg.reward_discount * next_potential - self.previous_potential
        )
        # Falling ends the episode; a negative terminal reward discourages it.
        total_reward -= 2.0 * self.reset_terminated.float()
        upright = tilt_error < 1.0 - math.cos(math.radians(10.0))
        height_ok = (self.chasis_pos[:, 2] -
                     self.reference_chasis_pos).abs() < 0.08
        moving_gently = (self.chasis_vel[:, 2].abs() < 0.25) & (
            self.imu_gyro.norm(dim=1) < 1.0)
        balanced = upright & height_ok & moving_gently
        self.upright_step_count.copy_(torch.where(
            balanced, self.upright_step_count + 1.0, 0.0))
        log = self.extras.setdefault("log", {})
        log["Metrics/orientation_error"] = tilt_error.mean()
        log["Metrics/tilt_degrees"] = torch.rad2deg(
            torch.acos((1.0 - tilt_error).clamp(-1.0, 1.0))
        ).mean()
        log["Metrics/upright_fraction"] = upright.float().mean()
        log["Metrics/balance_hold_seconds"] = self.upright_step_count.mean() * \
            self.step_dt
        return total_reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        # DirectRLEnv checks termination before it requests observations.
        self._read_measurements()
        tilt_error = compute_orientation_error(self.imu_quat)
        tilt_limit = 1.0 - math.cos(math.radians(self.cfg.max_tilt_degrees))
        out_of_bound = (tilt_error >= tilt_limit) | (
            self.chasis_pos[:, 2] < 0.1)
        # A fall on the last step is still a failure, not a bootstrapped timeout.
        time_out = (self.episode_length_buf >=
                    self.max_episode_length) & ~out_of_bound
        return out_of_bound, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        else:
            env_ids = torch.as_tensor(
                env_ids, dtype=torch.long, device=self.device
            )
        if env_ids.numel() == 0:
            return
        # Success requires ending a full episode with at least two consecutive
        # seconds of upright, settled motion after lifting from flat.
        survived = self.reset_time_outs[env_ids] & ~self.reset_terminated[env_ids]
        self.extras.setdefault("log", {})["Metrics/success_rate"] = (
            survived & (
                self.upright_step_count[env_ids] * self.step_dt >= self.cfg.balance_hold_seconds)
        ).float().mean()
        self.upright_step_count[env_ids] = 0.0
        self.previous_potential[env_ids] = 0.0

        # Clear the selected scene, sensor, actuator and episode buffers first.
        super()._reset_idx(env_ids)

        pose = self.car.data.default_root_pose.torch[env_ids].clone()
        # XYZW quaternion, pitch about Y. Place the wheel bottoms just above
        # the floor: wheel radius + rotated axle offset + 2 mm clearance.
        pitch = math.radians(self.cfg.reset_pitch_degrees) + (
            2.0 * torch.rand(env_ids.numel(), device=self.device) - 1.0
        ) * math.radians(self.cfg.reset_pitch_noise_degrees)
        pose[:, 2] = self.cfg.wheel_radius + \
            self.cfg.wheel_axle_offset * torch.sin(pitch) + 0.002
        pose[:, 3:7] = 0.0
        pose[:, 4] = torch.sin(pitch / 2.0)
        pose[:, 6] = torch.cos(pitch / 2.0)
        pose[:, :3] += self.scene.env_origins[env_ids]
        root_velocity = self.car.data.default_root_vel.torch[env_ids].clone()
        joint_pos = self.car.data.default_joint_pos.torch[env_ids].clone()
        joint_vel = self.car.data.default_joint_vel.torch[env_ids].clone()

        # Restore actual physics state, not just the measurement tensors.
        self.car.write_root_pose_to_sim_index(
            root_pose=pose,
            env_ids=env_ids,
        )
        self.car.write_root_velocity_to_sim_index(
            root_velocity=root_velocity,
            env_ids=env_ids,
        )
        self.car.write_joint_state_to_sim_index(
            position=joint_pos,
            velocity=joint_vel,
            env_ids=env_ids,
        )

        # Other cars keep their current actions and controller targets.
        self.actions[env_ids] = 0.0
        self.car.actuators.target_command.set_position_index(
            value=joint_pos[:, self.wheel_ids],
            joint_ids=self.wheel_ids,
            env_ids=env_ids,
        )
        self.car.actuators.target_command.set_velocity_index(
            value=joint_vel[:, self.wheel_ids],
            joint_ids=self.wheel_ids,
            env_ids=env_ids,
        )
        self._read_measurements()

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self._read_measurements()
        self.previous_potential.copy_(
            1.0 - 0.5 * compute_orientation_error(self.imu_quat))
        # Keep our own buffer so resetting it never changes the policy's tensor.
        self.actions.copy_(actions.clamp(-1.0, 1.0))

    def _apply_action(self) -> None:
        # Actions are [common drive, differential turn]. A common drive command
        # lets PPO discover coordinated lifting with both wheels together.
        drive = self.actions[:, 0]
        turn = self.cfg.turn_action_scale * self.actions[:, 1]
        wheel_commands = torch.stack(
            (drive + turn, drive - turn), dim=1).clamp(-1.0, 1.0)
        self.car.actuators.target_command.set_velocity_index(
            value=self.cfg.wheel_speed_limit * wheel_commands,
            joint_ids=self.wheel_ids,
        )

    def _read_measurements(self):
        # Wheel encoders.
        self.wheel_pos = self.car.data.joint_pos.torch[:, self.wheel_ids]
        self.wheel_vel = self.car.data.joint_vel.torch[:, self.wheel_ids]

        # IMU measurements, expressed in the IMU's local frame.
        self.imu_gyro = self.imu.data.ang_vel_b.torch
        self.imu_lin_accel = self.imu.data.lin_acc_b.torch

        # Simulator orientation and chassis COM position.
        self.imu_quat = self.car.data.root_link_quat_w.torch
        self.chasis_pos = self.car.data.root_com_pos_w.torch - self.scene.env_origins
        self.chasis_vel = self.car.data.root_com_lin_vel_w.torch


def compute_orientation_error(imu_quat):
    # XYZW: at upright, the chassis' negative local X axis points upward.
    # Dotting that axis with world up measures tilt independently of heading.
    x, y, z, w = imu_quat.unbind(dim=1)
    upright_alignment = 2.0 * (w * y - x * z)
    return (1.0 - upright_alignment).clamp(0.0, 2.0)  # 1 - cos(tilt)


def compute_rewards(chasis_pos, ref_chasis, imu_quat, imu_gyro, chasis_vel, actions, wheel_axle_offset=0.8):
    tilt_error = compute_orientation_error(imu_quat)
    lift_score = (1.0 - tilt_error).clamp_min(0.0).square()
    upright_score = torch.exp(-tilt_error / 0.04)
    # Predicted chassis height with the wheels grounded at this tilt. These
    # operands are all (N,): no mixing rewards across different cars.
    grounded_height = ref_chasis - wheel_axle_offset * tilt_error
    grounded_score = torch.exp(
        -((chasis_pos[:, 2] - grounded_height) / 0.05).square())
    # Flat or rear-on-floor postures earn almost no posture reward. Flying
    # through upright also earns less than lifting with the wheels grounded.
    posture_reward = (lift_score + upright_score) * grounded_score
    # Damping matters near upright; allow the rotation needed to lift first.
    rotation_cost = 0.0001 * lift_score * imu_gyro.square().sum(dim=1)
    bounce_cost = 0.002 * lift_score * chasis_vel[:, 2].square()
    action_cost = 0.0001 * actions.square().sum(dim=1)
    reward = posture_reward - rotation_cost - bounce_cost - action_cost
    return reward
