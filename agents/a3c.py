import gym
import time
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
from agents.base_agent import BaseAgent


class ActorCriticNet(nn.Module):
    def __init__(self, num_channel_input, num_output):
        super(ActorCriticNet, self).__init__()
        self.conv1 = nn.Conv2d(num_channel_input, 32, 3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.lstm = nn.LSTMCell(32 * 3 * 3, 256)
        self.critic_fc = nn.Linear(256, 1)
        self.actor_fc = nn.Linear(256, num_output)
        self.softmax = nn.Softmax()

    def forward(self, x, (h, c)):
        assert x.size(2) == 42 and x.size(3) == 42
        if h is None:
            h = Variable(torch.zeros(1, 256))
        if c is None:
            c = Variable(torch.zeros(1, 256))
        x = F.elu(self.conv1(x))
        x = F.elu(self.conv2(x))
        x = F.elu(self.conv3(x))
        x = F.elu(self.conv4(x))
        x = x.view(x.size(0), -1)
        h, c = self.lstm(x, (h, c))
        actor_prob = self.softmax(self.actor_fc(h))
        critic_value = self.critic_fc(h)
        return actor_prob, critic_value, (h, c)


class A3CAgent(BaseAgent):
    '''Asynchronous Advantage Actor Critic (A3C) agent.'''

    def __init__(self,
                 action_space,
                 observation_space,
                 learning_rate=1e-4,
                 discount=1.0,
                 t_max=20):
        if not isinstance(action_space, gym.spaces.Discrete):
            raise TypeError("Action space type should be Discrete.")
        self._action_space = action_space
        self._observation_space = observation_space
        self._learning_rate = learning_rate
        self._discount = discount
        self._t_max = t_max

        self._shared_model = ActorCriticNet(
            num_channel_input=observation_space.shape[0],
            num_output=action_space.n)
        self._shared_model.share_memory()
        self._slave_agent_list = []

        self.reset()

    def act(self, observation, greedy=False):
        prob, _, (self._lstm_h, self._lstm_c) = self._shared_model(
            Variable(torch.from_numpy(observation).unsqueeze(0)),
            (self._lstm_h, self._lstm_c))
        greedy_action = prob.max(1)[1].data
        if greedy:
            action = greedy_action
        else:
            action = prob.multinomial(1).data
        return action[0, 0]

    def create_async_learner(self):
        slave_agent = _A3CSlaveAgent(
            self._shared_model, self._action_space, self._observation_space,
            self._learning_rate, self._discount, self._t_max)
        return slave_agent

    def reset(self):
        self._lstm_h, self._lstm_c = None, None

    def learn(self, reward, observation, done):
        raise RuntimeError(
            "Not implemented. Please call create_slave_agent to "
            "generate async learners to perform the learning.")

    def save_model(self, model_path):
        torch.save(self._shared_model.state_dict(), model_path)

    def load_model(self, model_path):
        self._shared_model.load_state_dict(torch.load(model_path))


class _A3CSlaveAgent(BaseAgent):
    '''Asynchronous Advantage Actor Critic (A3C) agent.'''

    def __init__(self,
                 shared_model,
                 action_space,
                 observation_space,
                 learning_rate=1e-4,
                 discount=1.0,
                 t_max=20):
        if not isinstance(action_space, gym.spaces.Discrete):
            raise TypeError("Action space type should be Discrete.")
        self._shared_model = shared_model
        self._learning_rate = learning_rate
        self._discount = discount
        self._t_max = t_max

        self._local_model = ActorCriticNet(
            num_channel_input=observation_space.shape[0],
            num_output=action_space.n)
        self._optimizer = optim.Adam(
            self._shared_model.parameters(), lr=self._learning_rate)

        self._local_model.load_state_dict(self._shared_model.state_dict())
        self._lstm_h, self._lstm_c = None, None

        self._rewards = []
        self._values = []
        self._log_probs = []
        self._entropies = []

    def act(self, observation, greedy=False):
        prob, value, (self._lstm_h, self._lstm_c) = self._local_model(
            Variable(torch.from_numpy(observation).unsqueeze(0)),
            (self._lstm_h, self._lstm_c))
        greedy_action = prob.max(1)[1].data
        if greedy:
            action = greedy_action
        else:
            action = prob.multinomial(1).data
        self._action = action
        self._prob = prob
        self._value = value
        return action[0, 0]

    def reset(self):
        self._lstm_h, self._lstm_c = None, None

    def learn(self, reward, observation, done):
        log_prob = torch.log(self._prob)
        entropy = -(log_prob * self._prob).sum(1)
        action_log_prob = log_prob.gather(1, Variable(self._action))
        self._rewards.append(reward)
        self._values.append(self._value)
        self._log_probs.append(action_log_prob)
        self._entropies.append(entropy)

        if done or len(self._rewards) >= self._t_max:
            R = torch.zeros(1, 1)
            if not done:
                _, value, _ = self._local_model(
                    Variable(torch.from_numpy(observation).unsqueeze(0)),
                    (self._lstm_h, self._lstm_c))
                R = value.data
            self._values.append(Variable(R))
            R = Variable(R)
            gae = torch.zeros(1, 1)

            # compute n-step loss
            value_loss, policy_loss = 0, 0
            for i in reversed(range(len(self._rewards))):
                R = self._discount * R + self._rewards[i]
                advantage = R - self._values[i]
                value_loss += 0.5 * advantage.pow(2)
                # Generalized Advantage Estimataion
                delta_t = self._rewards[i] + self._discount * \
                    self._values[i + 1].data - self._values[i].data
                gae = gae * self._discount + delta_t
                policy_loss -= self._log_probs[i] * Variable(gae) \
                    + 0.01 * self._entropies[i]

            # compute grad and optimize
            self._optimizer.zero_grad()
            (policy_loss + 0.5 * value_loss).backward()
            torch.nn.utils.clip_grad_norm(self._local_model.parameters(), 40)
            self._share_grads(self._local_model)
            self._optimizer.step()

            # clean up
            self._rewards = []
            self._values = []
            self._log_probs = []
            self._entropies = []
            if done:
                self._lstm_h, self._lstm_c = None, None
            else:
                self._lstm_h = Variable(self._lstm_h.data)
                self._lstm_c = Variable(self._lstm_c.data)
                self._local_model.load_state_dict(
                    self._shared_model.state_dict())

    def _share_grads(self, model):
        for param, shared_param in zip(model.parameters(),
                                       self._shared_model.parameters()):
            if shared_param.grad is not None:
                return
            shared_param._grad = param.grad
