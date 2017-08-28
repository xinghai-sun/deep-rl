import cv2
import gym
from gym import spaces
import numpy as np


class AtariRescale42x42Wrapper(gym.ObservationWrapper):
    def __init__(self, env=None):
        super(AtariRescale42x42Wrapper, self).__init__(env)
        if isinstance(self.observation_space, spaces.Tuple):
            self.observation_space = spaces.Tuple([
                gym.spaces.Box(0.0, 1.0, [1, 42, 42])
                for space in self.env.observation_space.spaces
            ])
        else:
            self.observation_space = gym.spaces.Box(0.0, 1.0, [1, 42, 42])

    def _observation(self, observation):
        if isinstance(observation, tuple):
            return tuple([self._process_frame(obs) for obs in observation])
        else:
            return self._process_frame(observation)

    def _process_frame(self, frame):
        assert (frame.ndim == 3 and
                (frame.shape[2] == 3 or frame.shape[2] == 1) and
                frame.shape[0] == 210 and frame.shape[1] == 160)
        frame = frame[34:34 + 160, :160]
        frame = cv2.resize(frame, (42, 42))
        frame = frame.mean(2, dtype=np.float32)
        frame /= 255.0
        return np.expand_dims(frame, axis=0)


class NormalizeWrapper(gym.ObservationWrapper):
    def __init__(self, env=None):
        super(NormalizeWrapper, self).__init__(env)
        self._state_mean = 0
        self._state_std = 0
        self._alpha = 0.9999
        self._num_steps = 0

    def _observation(self, observation):
        if isinstance(observation, tuple):
            return tuple([self._normalize(obs) for obs in observation])
        else:
            return self._normalize(observation)

    def _normalize(self, observation):
        self._num_steps += 1
        self._state_mean = self._state_mean * self._alpha + \
            observation.mean() * (1 - self._alpha)
        self._state_std = self._state_std * self._alpha + \
            observation.std() * (1 - self._alpha)
        unbiased_mean = self._state_mean / (
            1 - pow(self._alpha, self._num_steps))
        unbiased_std = self._state_std / (
            1 - pow(self._alpha, self._num_steps))
        normalized_obs = (observation - unbiased_mean) / (unbiased_std + 1e-8)
        return normalized_obs
