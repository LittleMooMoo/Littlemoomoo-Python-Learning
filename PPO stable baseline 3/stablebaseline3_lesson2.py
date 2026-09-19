import gymnasium as gym
from stable_baselines3 import PPO
import os

model_dir = "models/PPO"
logdir = "logs"
TIMESTEPS = 10000

if not os.path.exists(model_dir):
    os.makedirs(model_dir)
if not os.path.exists(logdir):
    os.makedirs(logdir)

# make the lunar lander environment
env = gym.make("LunarLander-v3")
# create an A2C agent and train for a given number of environment steps
model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=logdir)


for i in range(1, 30):
    model.learn(total_timesteps=TIMESTEPS,
                reset_num_timesteps=False, tb_log_name="PPO")
    model.save(f"{model_dir}/{TIMESTEPS*i}")
# close the environment
env.close()
