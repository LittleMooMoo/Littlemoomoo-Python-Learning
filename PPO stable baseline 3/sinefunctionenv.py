import gymnasium as gym
import numpy as np


class SineEnv(gym.Env):
    def __init__(self):
        super(SineEnv,self).__init__()
        # the actions that can be taken, or "prediction"
        self.action_space = gym.spaces.Box(
            low = -2,
            high= 2,
            shape= (1,),
            dtype = np.float32
        )
        # the states that can be observed to affect the action, or "feedback"
        self.observation_space = gym.spaces.Box(
            low = -1,
            high= 1,
            shape = (1,),
            dtype = np.float32
        )
        self.angle = 0.0
        self.max_steps = 400
        self.angle_step = (2*np.pi)/self.max_steps
        self.steps = 0

    def reset(self,seed =None,options = None):
        super().reset(seed=seed)
        self.angle = 0
        self.steps = 0
        observation = np.array([self.angle/(2*np.pi)],dtype=np.float32)
        return observation,{}

    def step(self,action):
        prediction = float(action[0])
        correct_ans = float(np.sin(self.angle))

        reward = -(prediction - correct_ans)**2

        self.steps +=1
        self.angle += self.angle_step

        self.angle = self.angle % (2 * np.pi)
        observation = np.array([self.angle/(2*np.pi)],dtype=np.float32)
        terminated = False
        truncated = self.steps>=self.max_steps
        return observation,reward,terminated,truncated,{}