from stable_baselines3 import PPO
from sinefunctionenv import SineEnv
import matplotlib.pyplot as plt
import numpy as np

MAX_STEPS = 800

env = SineEnv()
model_dir = "/home/littlemoomoo/Littlemoomoo-Python-Learning/PPO stable baseline 3/trained_models"
model = PPO.load(f"{model_dir}/trainedsinmodel_v2",env = env)

step_numbers = []
rewards = []
actions =[]
observations = []
angles =[]

obs, info = env.reset()
for step_number in range(1,MAX_STEPS):
    action, _states = model.predict(obs, deterministic=True)
    actions.append(float(action[0]))
    observations.append(float(obs[0]))
    angles.append(float((obs[0]+1)*2*np.pi))
    obs, reward, terminated, truncated, info = env.step(action)
    step_numbers.append(step_number)
    rewards.append(float(reward))

    if terminated or truncated:
        obs, info = env.reset()
env.close()

plt.subplot(2,2,1)
plt.plot(step_numbers,rewards,color="red", linewidth=2, label="reward")
plt.title("reward vs steps")
plt.xlabel("steps")
plt.ylabel("reward")
plt.grid(True)


plt.subplot(2,2,2)
plt.plot(step_numbers,observations,color="blue", linewidth=2, label="observation")
plt.title("observations vs steps")
plt.xlabel("steps")
plt.ylabel("observations")
plt.grid(True)


plt.subplot(2,2,3)
plt.plot(step_numbers,actions,color="m", linewidth=2, label="actions")
plt.title("actions vs steps")
plt.xlabel("steps")
plt.ylabel("actions")
plt.grid(True)


plt.subplot(2,2,4)
plt.plot(step_numbers, np.sin(angles),
         color="black", linewidth=2, label="Correct sine")
plt.title("correct sine vs steps")
plt.xlabel("steps")
plt.ylabel("values")
plt.grid(True)

plt.show()