import gymnasium as gym
import numpy as np
from gymnasium import spaces
import numpy as np
from collections import deque
import cv2
import random
import time

SNAKE_LEN_GOAL = 30

def collision_with_apple(apple_position, score):
    apple_position = [random.randrange(1, 50)*10, random.randrange(1, 50)*10]
    score += 100
    return apple_position, score


def collision_with_boundaries(snake_head):
    if snake_head[0] >= 500 or snake_head[0] < 0 or snake_head[1] >= 500 or snake_head[1] < 0:
        return 1
    else:
        return 0


def collision_with_self(snake_position):
    snake_head = snake_position[0]
    if snake_head in snake_position[1:]:
        return 1
    else:
        return 0


class SnakeEnv(gym.Env):
    """Custom Environment that follows gym interface."""

    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self):
        super(SnakeEnv).__init__()
        # Define action and observation space
        # They must be gym.spaces objects
        # Example when using discrete actions:
        self.action_space = spaces.Discrete(4)
        # Example for using image as input (channel-first; channel-last also works):
        self.observation_space = spaces.Box(low=-np.inf,
                                            high=np.inf,
                                            shape=(5 + SNAKE_LEN_GOAL,),
                                            dtype=np.float32)

    def step(self, action):
        while True:
            self.previous_len = self.snake_length
            # Change the head position based on the button direction
            if action == 1:
                self.snake_head[0] += 10
            elif action == 0:
                self.snake_head[0] -= 10
            elif action == 2:
                self.snake_head[1] += 10
            elif action == 3:
                self.snake_head[1] -= 10

            # Increase Snake length on eating apple
            if self.snake_head == self.apple_position:
                self.apple_position, self.score = collision_with_apple(
                    self.apple_position, self.score)
                self.snake_position.insert(0, list(self.snake_head))

            else:
                self.snake_position.insert(0, list(self.snake_head))
                self.snake_position.pop()

            # On collision kill the snake and print the score
            if collision_with_boundaries(self.snake_head) == 1 or collision_with_self(self.snake_position) == 1:
                self.font = cv2.FONT_HERSHEY_SIMPLEX
                self.img = np.zeros((500, 500, 3), dtype='uint8')
                self.done = True
                self.head_x = self.snake_head[0]
                self.head_y = self.snake_head[1]

                self.apple_delta_x = self.head_x - self.apple_position[0]
                self.apple_delta_y = self.head_y - self.apple_position[1]
                self.snake_length = len(self.snake_position)

            self.reward = self.score
            if self.done:
                self.reward = -10.0
            else:
                self.reward = self.reward + 0.1

            self.head_x, self.head_y = self.snake_head
            self.apple_delta_x = self.head_x - self.apple_position[0]
            self.apple_delta_y = self.head_y - self.apple_position[1]
            self.snake_length = len(self.snake_position)

            self.delta_snake_length = self.snake_length - self.previous_len

            self.dist_to_apple = float(np.sqrt(self.apple_delta_x**2 + self.apple_delta_y**2))

            self.reward = self.reward - 0.1 * self.dist_to_apple + self.delta_snake_length

            # Record one action. The deque keeps the most recent 30.
            self.prev_actions.append(int(action))

            self.observation = np.array(
                [
                    self.head_x,
                    self.head_y,
                    self.apple_delta_x,
                    self.apple_delta_y,
                    self.snake_length,
                ] + list(self.prev_actions),
                dtype=np.float32,
            )
            return self.observation, float(self.reward), bool(self.done), False, self.info

    def reset(self, *, seed=None, options=None):
        self.done = False
        self.img = np.zeros((500, 500, 3), dtype='uint8')
        # Initial Snake and Apple position
        self.snake_position = [[250, 250], [240, 250], [230, 250]]
        self.apple_position = [random.randrange(
            1, 50)*10, random.randrange(1, 50)*10]
        self.score = 0
        self.reward = 0
        self.prev_button_direction = 1
        self.button_direction = 1
        self.snake_head = [250, 250]

        self.info = {}
        # head x, head y, apple x, apple y, snake length, previous moves

        self.head_x = self.snake_head[0]
        self.head_y = self.snake_head[1]

        self.apple_delta_x = self.head_x - self.apple_position[0]
        self.apple_delta_y = self.head_y - self.apple_position[1]

        self.dist_to_apple = float(np.sqrt(self.apple_delta_x**2 + self.apple_delta_y**2))

        self.snake_length = len(self.snake_position)
        self.prev_actions = deque(maxlen=SNAKE_LEN_GOAL)


        for _ in range(SNAKE_LEN_GOAL):
            self.prev_actions.append(-1)
            self.observation = np.array([
                self.head_x,
                self.head_y,
                self.apple_delta_x,
                self.apple_delta_y,
                self.snake_length
            ] + list(self.prev_actions),
                dtype=np.float32,
            )

        return self.observation, self.info

    def render(self)->None:
        self.img = np.zeros((500, 500, 3), dtype='uint8')
        # Display Apple
        cv2.rectangle(self.img, (self.apple_position[0], self.apple_position[1]),
                          (self.apple_position[0]+10, self.apple_position[1]+10), (0, 0, 255), 3)
        # Display Snake
        for position in self.snake_position:
            cv2.rectangle(self.img, (position[0], position[1]),(position[0]+10, position[1]+10), (0, 255, 0), 3)

        cv2.imshow('a', self.img)
        cv2.waitKey(33)
