import gymnasium as gym
import torch

print(torch.rand(5, 3))

# Create the simulation.
env = gym.make("LunarLander-v3")
# For continuous control:
# env = gym.make("LunarLander-v3", continuous=True)

# Reset returns the initial observation and additional information.
observation, info = env.reset()

print("sample action:", env.action_space.sample())
print("observation space shape:", env.observation_space.shape)
print("initial observation:", observation)

# Apply one random action to the simulation.
action = env.action_space.sample()
observation, reward, terminated, truncated, info = env.step(action)

print("observation after action:", observation)
print("reward:", reward)

env.close()
