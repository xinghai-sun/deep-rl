from abc import ABCMeta, abstractmethod


class BaseAgent(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def __init__(self, action_space, observation_space):
        pass

    @abstractmethod
    def act(self, observation, greedy=False):
        pass

    @abstractmethod
    def learn(self, reward, next_observation, done):
        pass

    def reset(self):
        pass
