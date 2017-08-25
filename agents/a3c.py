import gym
import time
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
from torch.autograd import Variable
import torch.optim as optim


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


class A3CAgent(object):
    '''Asynchronous Advantage Actor Critic (A3C) agent.'''

    def __init__(self,
                 action_space,
                 observation_space,
                 learning_rate=1e-4,
                 discount=1.0,
                 t_max=20,
                 save_to_team=None):
        if not isinstance(action_space, gym.spaces.Discrete):
            raise TypeError("Action space type should be Discrete.")
        self._learning_rate = learning_rate
        self._discount = discount
        self._t_max = t_max
        self._save_to_team = save_to_team
        self.reset()

        self._shared_model = ActorCriticNet(
            num_channel_input=observation_space.shape[0],
            num_output=action_space.n)
        self._shared_model.share_memory()

        if self._save_to_team is not None:
            self._save_to_team.append(copy.deepcopy(self))

    def act(self, observation):
        prob, value, (self._lstm_h, self._lstm_c) = self._shared_model(
            Variable(torch.from_numpy(observation).unsqueeze(0)),
            (self._lstm_h, self._lstm_c))
        _, action = prob[0].data.max(0)
        return action[0]

    def reset(self):
        self._lstm_h, self._lstm_c = None, None

    def learn_async(self, env_generator, num_processes, enable_test=True):
        self._env_generator = env_generator
        processes = []
        if enable_test:
            p = mp.Process(target=self._async_tester, args=(num_processes, ))
            p.start()
            processes.append(p)
        for process_id in range(0, num_processes):
            p = mp.Process(target=self._async_learner, args=(process_id, ))
            p.start()
            processes.append(p)
        for p in processes:
            p.join()

    def _async_learner(self, process_id):
        # prepare local env and optimizer
        torch.manual_seed(process_id)
        env = self._env_generator()
        env.seed(process_id)
        optimizer = optim.Adam(
            self._shared_model.parameters(), lr=self._learning_rate)
        # prepare local model
        model = ActorCriticNet(
            num_channel_input=env.observation_space.shape[0],
            num_output=env.action_space.n)
        # explore and learn
        observation = env.reset()
        done = True
        cum_rewards = 0
        num_async_update = 0
        while True:
            model.load_state_dict(self._shared_model.state_dict())
            if done:
                lstm_h, lstm_c = None, None
            else:
                lstm_h = Variable(lstm_h.data)
                lstm_c = Variable(lstm_c.data)

            rewards, values, log_probs, entropies = [], [], [], []
            # take n-step actions
            for step in xrange(self._t_max):
                prob, value, (lstm_h, lstm_c) = model(
                    Variable(torch.from_numpy(observation).unsqueeze(0)),
                    (lstm_h, lstm_c))
                log_prob = torch.log(prob)
                entropy = -(log_prob * prob).sum(1)
                action = prob.multinomial(1).data
                action_log_prob = log_prob.gather(1, Variable(action))
                observation, reward, done, _ = env.step(action[0, 0])

                rewards.append(reward)
                values.append(value)
                log_probs.append(action_log_prob)
                entropies.append(entropy)

                cum_rewards += reward
                if done:
                    observation = env.reset()
                    if process_id == 0:
                        print(cum_rewards)
                    cum_rewards = 0
                    break

            R = torch.zeros(1, 1)
            if not done:
                _, value, _ = model(
                    Variable(torch.from_numpy(observation).unsqueeze(0)),
                    (lstm_h, lstm_c))
                R = value.data
            values.append(Variable(R))
            R = Variable(R)
            gae = torch.zeros(1, 1)

            # n-step loss
            value_loss, policy_loss = 0, 0
            for i in reversed(range(len(rewards))):
                R = self._discount * R + rewards[i]
                advantage = R - values[i]
                value_loss += 0.5 * advantage.pow(2)
                # Generalized Advantage Estimataion
                delta_t = rewards[i] + self._discount * values[i + 1].data \
                    - values[i].data
                gae = gae * self._discount + delta_t
                policy_loss -= log_probs[i] * Variable(gae) \
                    + 0.01 * entropies[i]

            optimizer.zero_grad()
            total_loss = policy_loss + 0.5 * value_loss
            total_loss.backward()
            torch.nn.utils.clip_grad_norm(model.parameters(), 40)
            self._share_grads(model)
            optimizer.step()
            optimizer.step()

            if self._save_to_team is not None and num_async_update % 10 == 0:
                self._save_to_team.append(copy.deepcopy(self))
            num_async_update += 1
            print(agent_team)

        env.close()

    def _async_tester(self, process_id):
        # prepare local env and optimizer
        torch.manual_seed(process_id)
        env = self._env_generator()
        env.seed(process_id)
        # prepare local model
        model = ActorCriticNet(
            num_channel_input=env.observation_space.shape[0],
            num_output=env.action_space.n)
        model.eval()
        # explore and learn
        model.load_state_dict(self._shared_model.state_dict())
        lstm_h, lstm_c = None, None
        observation = env.reset()
        done = True
        cum_rewards = 0
        time_start = time.time()
        while True:
            prob, _, (lstm_h, lstm_c) = model(
                Variable(torch.from_numpy(observation).unsqueeze(0)),
                (lstm_h, lstm_c))
            _, action = prob[0].data.max(0)
            observation, reward, done, _ = env.step(action[0])
            cum_rewards += reward
            if done:
                model.load_state_dict(self._shared_model.state_dict())
                lstm_h, lstm_c = None, None
                observation = env.reset()
                print('[Time=%.1f min] Test Cummutive Rewards: %f' % (
                    (time.time() - time_start) / 60.0, cum_rewards))
                cum_rewards = 0
                time.sleep(60)
        env.close()

    def _share_grads(self, model):
        for param, shared_param in zip(model.parameters(),
                                       self._shared_model.parameters()):
            if shared_param.grad is not None:
                return
            shared_param._grad = param.grad
