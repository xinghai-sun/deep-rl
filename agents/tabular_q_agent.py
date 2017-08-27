from collections import defaultdict
import numpy as np
from gym import spaces
import copy
from agents.base_agent import BaseAgent


class TabularQAgent(BaseAgent):
    '''Tabular Q-learning agent.'''

    def __init__(self,
                 action_space,
                 observation_space,
                 q_init=0.0,
                 learning_rate=0.1,
                 discount=1.0,
                 epsilon=0.05):
        if not isinstance(action_space, spaces.Discrete):
            raise TypeError("Action space type should be Discrete.")
        if not isinstance(observation_space, spaces.Discrete):
            raise TypeError("Observation space type should be Discrete.")
        self._action_space = action_space
        self._learning_rate = learning_rate
        self._discount = discount
        self._epsilon = epsilon
        self._q = defaultdict(lambda: q_init * np.ones(action_space.n))

    def act(self, observation, greedy=False):
        greedy_action = np.argmax(self._q[observation])
        if greedy or np.random.random() >= self._epsilon:
            action = greedy_action
        else:
            action = self._action_space.sample()
        self._observation = observation
        self._action = action
        return action

    def learn(self, reward, next_observation, done):
        future = np.max(self._q[next_observation]) if not done else 0.0
        before = self._q[self._observation][self._action]
        target = self._learning_rate * (reward + self._discount * future - \
            self._q[self._observation][self._action])
        self._q[self._observation][self._action] += self._learning_rate * (
            reward + self._discount * future - \
                self._q[self._observation][self._action])
