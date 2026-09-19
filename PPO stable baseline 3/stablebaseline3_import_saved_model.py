import gymnasium as gym
from stable_baselines3 import PPO


model_dir = "models/PPO"
logdir = "logs"
model_path = f"{model_dir}/17000.zip"

env = gym.make("LunarLander-v3", render_mode="human")

model = PPO.load("ppo_lunarlander", env=env)

for ep in range(10):
    observation, info = env.reset()
    done = False
    total_reward = 0

    while not done:
        action, _ = model.predict(observation, deterministic=True)

        observation, reward, terminated, truncated, info = env.step(action)

        total_reward += reward
        done = terminated or truncated

    print(f"Episode {ep + 1}: total reward = {total_reward:.1f}")

env.close()
