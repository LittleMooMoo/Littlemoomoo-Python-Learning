from stable_baselines3 import PPO
from sinefunctionenv import SineEnv

KEEPTRAINING = False
TOTAL_EPOCH = 20
TIMESTEPS = 10000
LEARNING_RATE = 0.0003


env = SineEnv()
env.reset()

model_dir = "/home/littlemoomoo/Littlemoomoo-Python-Learning/PPO stable baseline 3/trained_models"

if KEEPTRAINING == True:
    model = PPO.load(f"{model_dir}/trainedsinmodel_v2",env= env)
else:
    model = PPO('MlpPolicy', env,gamma=0.0,learning_rate=LEARNING_RATE, verbose=1)

for i in range(TOTAL_EPOCH):
    model.learn(total_timesteps=TIMESTEPS,reset_num_timesteps=False)
    model.save(f"{model_dir}/trainedsinmodel_v2")
env.close()
