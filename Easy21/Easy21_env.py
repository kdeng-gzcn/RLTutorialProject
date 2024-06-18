import gymnasium as gym
import numpy as np
from gymnasium import spaces\

import random


def hit(current_points, is_player=True):
    points = random.randint(1, 10)  # int

    if is_player:
        if random.random() < 1 / 3:  # red
            current_points -= points
        else:  # black
            current_points += points
    else:
        current_points += points

    return current_points

class Easy21(gym.Env):
    """Custom Environment that follows gym interface."""

    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self):
        super().__init__()
        # Define action and observation space
        # They must be gym.spaces objects
        # Example when using discrete actions:
        self.action_space = spaces.Discrete(2)
        # Example for using image as input (channel-first; channel-last also works):
        self.observation_space = spaces.Box(low=-50, high=50,
                                            shape=(2,), dtype=np.int64)

    def step(self, action):
        if action == 0: # hit
            self.player_points = hit(self.player_points, is_player=True)
            if self.player_points < 1 or self.player_points > 21:
                self.reward = -1
                self.terminated = True

        if action == 1: # stick
            # dealer
            while 1 < self.dealer_points and self.dealer_points < 17:
                self.dealer_points = hit(self.dealer_points, is_player=False)
                if self.dealer_points < 1 or self.dealer_points > 21:
                    self.reward = 1
                    self.terminated = True

            # end the game
            if self.player_points > self.dealer_points:
                self.reward = 1
                self.terminated = True
            elif self.player_points == self.dealer_points:
                self.reward = 0
                self.terminated = True
            else:
                self.reward = -1
                self.terminated = True


        # obs
        player_points_now = self.player_points
        dealer_points_now = self.dealer_points
        self.observation = np.array([player_points_now - dealer_points_now, 21 - player_points_now])

        self.truncated = False
        self.info = {}
        return self.observation, self.reward, self.terminated, self.truncated, self.info

    def reset(self, seed=None, options=None):
        self.player_points = 0
        self.dealer_points = 0
        self.reward = 0
        self.terminated = False

        # initial
        self.player_points = hit(self.player_points, is_player=False) # must be black
        self.dealer_points = hit(self.dealer_points, is_player=False)

        # obs
        player_points_now = self.player_points
        dealer_points_now = self.dealer_points
        self.observation = np.array([player_points_now - dealer_points_now, 21 - player_points_now])

        self.info = {}
        return self.observation, self.info




