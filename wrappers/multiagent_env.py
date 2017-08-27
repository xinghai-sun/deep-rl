import cv2
import gym
import numpy as np


class AddOneOpponentWrapper(gym.Wrapper):
    def __init__(self, env, opponent_candidates):
        super(AddOneOpponentWrapper, self).__init__(env)
        self.observation_space = self.env.observation_space[0]
        self.action_space = self.env.observation_space[0]

    def _step(self, action):
        opponent_action = self._opponent_actor.act(self._opponent_obs)
        obs, reward, done, info = self.env.step([action, opponent_action])
        self._opponent_obs = obs[1]
        return obs[0], reward[0], done, info

    def _reset(self):
        obs = self.env.reset()
        self._opponent_actor = opponent_candidates.choice()
        self._opponent_obs = obs[1]
        return obs[0]
