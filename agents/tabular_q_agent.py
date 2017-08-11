from collections import defaultdict
import numpy as np
from gym import spaces
import copy


class TabularQAgent(object):
    '''Tabular Q-learning agent.'''

    def __init__(self,
                 action_space,
                 observation_space,
                 q_init=0.0,
                 learning_rate=0.1,
                 discount=1.0,
                 epsilon=0.01):
        if not isinstance(action_space, spaces.Discrete):
            raise TypeError("Action space type should be Discrete.")
        if not isinstance(observation_space, spaces.Discrete):
            raise TypeError("Observation space type should be Discrete.")
        self._action_space = action_space
        self._learning_rate = learning_rate
        self._discount = discount
        self._epsilon = epsilon
        self._q = defaultdict(lambda: q_init * np.ones(action_space.n))

    def act(self, observation):
        if np.random.random() >= self._epsilon:
            return np.argmax(self._q[observation])
        else:
            return self._action_space.sample()

    def learn(self, observation, action, reward, next_observation, done):
        future = np.max(self._q[next_observation]) if not done else 0.0
        before = self._q[observation][action]
        target = self._learning_rate * (
            reward + self._discount * future - self._q[observation][action])
        self._q[observation][action] += self._learning_rate * (
            reward + self._discount * future - self._q[observation][action])
