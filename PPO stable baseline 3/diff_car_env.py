from gymnasium import spaces
import gymnasium as gym
import mujoco
import numpy as np
from pathlib import Path
import warp as wp
import mujoco_warp as mjw

xml_path = Path(__file__).resolve().parent/"diff_car.xml"
car = mujoco.MjModel.from_xml_path(str(xml_path))


class DiffCarEnv(gym.Env):
    def __init__(self):
        self.action_space = spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Box(
            -np.inf, np.inf, shape=(8,), dtype=np.float32
        )
