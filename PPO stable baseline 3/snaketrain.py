from stable_baselines3 import PPO
import os
from snakegameenvconvert import SnakeEnv

CONTINUE_TRAINING = False

models_dir = "/home/littlemoomoo/Littlemoomoo-Python-Learning/PPO stable baseline 3/trained_models"

if not os.path.exists(models_dir):
    os.makedirs(models_dir)

env = SnakeEnv()
env.reset()

if CONTINUE_TRAINING == True:
    model = PPO.load(f"{models_dir}/snakegame_modelv1",env= env)
else:
    model = PPO('MlpPolicy', env, verbose=1)


TIMESTEPS = 10000


for i in range(1,20):
    model.learn(total_timesteps=TIMESTEPS,
                reset_num_timesteps=False, tb_log_name="PPO")
    model.save(f"{models_dir}/snakegame_modelv1")
env.close()
