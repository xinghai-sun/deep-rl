from scipy.misc import imresize
from skimage import color
import gym
from gym import spaces
import numpy as np


class RescaleFrameWrapper(gym.Wrapper):
    def __init__(self, env, target_size=(84, 84), grayscale=True):
        super(RescaleFrameWrapper, self).__init__(env)
        self._target_size = target_size
        self._grayscale = grayscale
        channel = 1 if grayscale else self.env.observation_space.shape[2]
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(target_size[0], target_size[1], channel))

    def _step(self, action):
        obs, reward, done, info = self.env.step(action)
        return self._observation(obs), reward, done, info

    def _reset(self):
        obs = self.env.reset()
        return self._observation(obs)

    def _observation(self, obs):
        assert obs.ndim == 3 and (obs.shape[2] == 3 or obs.shape[2] == 1)
        obs = imresize(obs, self._target_size)
        if self._grayscale and obs.shape[2] == 3:
            obs = np.expand_dims(color.rgb2gray(obs), 2)
        return obs


class TransposeNormalizeWrapper(gym.Wrapper):
    def __init__(self, env):
        super(TransposeNormalizeWrapper, self).__init__(env)
        obs_shape = self.env.observation_space.shape
        self.observation_space = spaces.Box(
            low=0, high=1.0, shape=(obs_shape[2], obs_shape[0], obs_shape[1]))

    def _step(self, action):
        obs, reward, done, info = self.env.step(action)
        return self._observation(obs), reward, done, info

    def _reset(self):
        obs = self.env.reset()
        return self._observation(obs)

    def _observation(self, obs):
        assert obs.ndim == 3
        return np.transpose(obs / 255.0, (2, 0, 1))
