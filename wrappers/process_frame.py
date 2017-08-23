from scipy.misc import imresize
from skimage import color
import gym
from gym import spaces
import numpy as np


class ResizeFrameWrapper(gym.Wrapper):
    def __init__(self, env, target_size=(84, 84), grayscale=True):
        super(ResizeFrameWrapper, self).__init__(env)
        self._target_size = target_size
        self._grayscale = grayscale
        channel = 1 if grayscale else self.observation_space.shape[2]
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(target_size[0], target_size[1], channel))

    def _step(self, action):
        obs, reward, done, info = self.env.step(action)
        assert obs.ndim == 3 and (obs.shape[2] == 3 or obs.shape[2] == 1)
        obs = imresize(obs, self._target_size)
        if self._grayscale and obs.shape[2] == 3:
            obs = np.expand_dims(color.rgb2gray(obs), 2)
        return obs, reward, done, info

    def _reset(self):
        obs = self.env.reset()
        assert obs.ndim == 3 and (obs.shape[2] == 3 or obs.shape[2] == 1)
        obs = imresize(obs, self._target_size)
        if self._grayscale and obs.shape[2] == 3:
            obs = np.expand_dims(color.rgb2gray(obs), 2)
        return obs
