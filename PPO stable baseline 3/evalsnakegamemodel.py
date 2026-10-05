from stable_baselines3 import PPO
from snakegameenvconvert import SnakeEnv

models_dir = "/home/littlemoomoo/Littlemoomoo-Python-Learning/PPO stable baseline 3/trained_models"
logdir = "/home/littlemoomoo/Littlemoomoo-Python-Learning/PPO stable baseline 3/tensor_board_logs"

env = SnakeEnv()

model = PPO.load(f"{models_dir}/snakegame_modelv1", env=env)

for ep in range(100):
    observation, info = env.reset()
    env.render()
    done = False
    total_reward = 0

    while not done:
        action, _ = model.predict(observation, deterministic=True)

        observation, reward, terminated, truncated, info = env.step(action)
        env.render()

        total_reward += reward
        done = terminated or truncated

    print(f"Episode {ep + 1}: total reward = {total_reward:.1f}")

env.close()
