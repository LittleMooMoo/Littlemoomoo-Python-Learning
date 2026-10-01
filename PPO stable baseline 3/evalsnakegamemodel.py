from stable_baselines3 import PPO
from snakegameenvconvert import SnakeEnv

model_dir = "models/PPO"
logdir = "logs"
model_path = f"{model_dir}/17000.zip"

env = SnakeEnv()

model = PPO.load("checksnakeenv", env=env)

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
