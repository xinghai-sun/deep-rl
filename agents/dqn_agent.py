import copy
import numpy as np
import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
from gym import wrappers
from gym import spaces
from agents.base_agent import BaseAgent
from agents.memory import ReplayMemory, Transition


class FCNet(nn.Module):
    def __init__(self, input_size, output_size):
        super(FCNet, self).__init__()
        self.fc1 = nn.Linear(input_size, 1024)
        self.fc2 = nn.Linear(1024, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class DQNAgent(BaseAgent):
    '''Deep Q-learning agent.'''

    def __init__(self,
                 action_space,
                 observation_space,
                 batch_size=128,
                 learning_rate=1e-3,
                 discount=1.0,
                 epsilon=0.05):
        if not isinstance(action_space, spaces.Discrete):
            raise TypeError("Action space type should be Discrete.")
        self._action_space = action_space
        self._batch_size = batch_size
        self._discount = discount
        self._epsilon = epsilon
        self._q_network = FCNet(
            input_size=reduce(lambda x, y: x * y, observation_space.shape),
            output_size=action_space.n)
        self._optimizer = optim.RMSprop(
            self._q_network.parameters(), lr=learning_rate)
        self._memory = ReplayMemory(100000)

    def act(self, observation, greedy=False):
        q_values = self._q_network(
            Variable(torch.FloatTensor([observation]).view(1, -1)))
        _, action = q_values[0].data.max(0)
        greedy_action = action[0]
        if greedy or np.random.random() >= self._epsilon:
            action = greedy_action
        else:
            action = self._action_space.sample()
        self._observation = observation
        self._action = action
        return action

    def learn(self, reward, next_observation, done):
        # experience replay
        self._memory.push(self._observation, self._action, reward,
                          next_observation, done)
        if len(self._memory) < self._batch_size:
            return
        transitions = self._memory.sample(self._batch_size)
        batch = Transition(*zip(*transitions))
        # convert to torch variable
        next_observation_batch = Variable(
            torch.FloatTensor(batch.next_observation).view(self._batch_size,
                                                           -1),
            volatile=True)
        observation_batch = Variable(
            torch.FloatTensor(batch.observation).view(self._batch_size, -1))
        reward_batch = Variable(torch.FloatTensor(batch.reward))
        action_batch = Variable(torch.LongTensor(batch.action))
        done_batch = Variable(torch.Tensor(batch.done))
        # compute max-q target
        q_values_next = self._q_network(next_observation_batch)
        futures = q_values_next.max(dim=1)[0] * (1 - done_batch)
        target_q = reward_batch + self._discount * futures
        target_q.volatile = False
        # compute gradient
        q_values = self._q_network(observation_batch)
        loss_fn = torch.nn.MSELoss()
        loss = loss_fn(q_values.gather(1, action_batch.view(-1, 1)), target_q)
        self._optimizer.zero_grad()
        loss.backward()
        # update q-network
        self._optimizer.step()
