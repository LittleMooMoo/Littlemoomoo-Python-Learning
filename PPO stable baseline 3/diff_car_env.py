import mujoco
import numpy as np
from pathlib import Path
import warp as wp
import mujoco_warp as mjw
import torch
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab_newton.physics import NewtonCfg, MJWarpSolverCfg
from collections.abc import Sequence
from isaaclab.assets import ArticulationCfg
from isaaclab.actuators import ImplicitActuatorCfg

# Similar to the manager-based environments, a configuration class is defined
# for the task to hold settings for the simulation parameters, the scene, the actors, and the task.

CAR_XML_PATH = Path(__file__).resolve().parent/"diff_car.xml"


@configclass
class DiffCarEnvCfg(DirectRLEnvCfg):
    action_space = 4
    observation_space = 14
    state_space = 0
    orient_error_max = 0.30
    episode_length_buf = 1
    decimation = 2
    episode_length_s = 20.0
    sim = sim_utils.SimulationCfg(
        dt=1 / 120,
        physics=NewtonCfg(solver_cfg=MJWarpSolverCfg()),
    )
    scene = DiffCarSceneCfg(
        num_envs=16,
        env_spacing=4.0,
    )


@configclass
class DiffCarSceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(prim_path="/World/ground",
                          spawn=sim_utils.GroundPlaneCfg())
    car = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Car",
        spawn=sim_utils.MjcfFileCfg(
            asset_path=str(CAR_XML_PATH),
            fix_base=False,
        ),
        actuators={
            "wheels": ImplicitActuatorCfg(
                joint_names_expr=["left_axle", "right_axle"],
                stiffness=20.0,
                damping=2.0,
            ),
        },
    )


class DiffCarEnv(DirectRLEnv):
    cfg: DiffCarEnvCfg

    def __init__(self, cfg: DiffCarEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self.car = self.scene["car"]
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

    def _get_observations(self) -> dict:
        # dim select which axis to concatenate along,
        # dim=1 is row, dim=2 is columns
        obs = torch.cat(
            (self.wheel_pos,
             self.wheel_vel,
             self.imu_gyro,
             self.imu_lin_accel,
             self.imu_quat), dim=1
        )
        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        total_reward = compute_rewards(self.chasis_pos,
                                       self.imu_quat,
                                       self.imu_gyro,
                                       self.wheel_vel)
        return total_reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        ref_q = torch.tensor([0, 0.7071, 0, 0.7071],
                             dtype=torch.float32, device=self.imu_quat.device)

        ref_q_normalized = ref_q / torch.linalg.norm(ref_q)
        alignment = (ref_q_normalized * self.imu_quat).sum(dim=1)
        q_error = (1 - alignment.square()).clamp_min(0)
        out_of_bound = q_error >= self.cfg.orient_error_max
        time_out = self.episode_length_buf >= self.max_episode_length
        return out_of_bound, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.car._ALL_INDICES
        # log the survival success rate before resetting (survived = timed out without terminating early)
        survived = self.reset_time_outs[env_ids].float()
        # no .item(): avoids a sync on every reset
        self.extras.setdefault("log", {})[
            "Metrics/success_rate"] = survived.mean()
        pose = self.car.data.default_root_pose.torch[env_ids].clone()
        # Each parallel environment has its own starting location.
        pose[:, :3] += self.scene.env_origins[env_ids]
        self.car.write_root_pose_to_sim_index(
            root_pose=pose,
            env_ids=env_ids,
        )
        super()._reset_idx(env_ids)


def compute_rewards(chasis_pos,
                    imu_quat,
                    imu_gyro,
                    wheel_vel):

    balancing_score = \
        0.5 * (chasis_pos[:, 2] >= 0.55).float() - \
        0.5 * imu_gyro.square().sum(dim=1)

    control_effort = wheel_vel.square().sum(dim=1)

    ref_q = torch.tensor([0, 0.7071, 0, 0.7071],
                         dtype=torch.float32, device=imu_quat.device)

    ref_q_normalized = ref_q / torch.linalg.norm(ref_q)

    alignment = (ref_q_normalized * imu_quat).sum(dim=1)
    q_error = (1 - alignment.square()).clamp_min(0)
    reward = balancing_score - \
        0.0001 * control_effort - 0.1 * q_error
    return reward
