import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
from torch.autograd import Variable
import torch.optim as optim


class ActorCriticNet(nn.Module):
    def __init__(self, input_channel, policy_output_size):
        super(ActorCriticNet, self).__init__()
        self.conv1 = nn.Conv2d(input_channel, 16, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=4, stride=2)
        self.policy_fc = nn.Linear(32 * 9 * 9, policy_output_size)
        self.value_fc = nn.Linear(32 * 9 * 9, 1)
        self.softmax = nn.Softmax()
        self.train()

    def forward(self, x):
        assert x.size(2) == 84 and x.size(3) == 84
        x = x.float()
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        policy = self.softmax(self.policy_fc(x))
        value = self.value_fc(x)
        return policy, value


class A3CAgent(object):
    '''Deep Q-learning agent.'''

    def __init__(self,
                 env_generator,
                 learning_rate=1e-3,
                 discount=1.0,
                 t_max=20):
        self._env_generator = env_generator
        self._learning_rate = learning_rate
        self._discount = discount
        self._t_max = t_max

        env = env_generator()
        self._shared_model = ActorCriticNet(
            input_channel=env.observation_space.shape[0],
            policy_output_size=env.action_space.n)
        self._shared_model.share_memory()

    def act(self, observation):
        policy, value = self._shared_model(
            Variable(torch.from_numpy(observation).unsqueeze(0)))
        _, action = policy[0].data.max(0)
        return action[0]

    def learn(self, num_processes):
        processes = []
        for process_id in range(0, num_processes):
            p = mp.Process(target=self._single_learner, args=(process_id, ))
            p.start()
            processes.append(p)
        for p in processes:
            p.join()

    def _single_learner(self, process_id):
        torch.manual_seed(process_id)
        # prepare env
        env = self._env_generator()
        env.seed(process_id)
        # prepare optimizer
        optimizer = optim.RMSprop(
            self._shared_model.parameters(), lr=self._learning_rate)
        # prepare local model
        model = ActorCriticNet(
            input_channel=env.observation_space.shape[0],
            policy_output_size=env.action_space.n)
        model.train()
        # explore and learn
        observation = env.reset()
        done = True
        cum_rewards = 0

        while True:
            model.load_state_dict(self._shared_model.state_dict())
            rewards, values, log_policies, entropies = [], [], [], []
            for step in xrange(self._t_max):
                policy, value = model(
                    Variable(torch.from_numpy(observation).unsqueeze(0)))
                log_policy = torch.log(policy)
                action = policy.multinomial(1).data
                action_log_prob = log_policy.gather(1, Variable(action))
                observation, reward, done, _ = env.step(action[0, 0])
                entropy = -(log_policy * policy).sum(1)
                rewards.append(reward)
                values.append(value)
                log_policies.append(action_log_prob)
                entropies.append(entropy)
                cum_rewards += reward
                if done:
                    print(cum_rewards)
                    cum_rewards = 0
                    observation = env.reset()
                    break
            value_loss, policy_loss = 0, 0
            if not done:
                _, value = model(
                    Variable(torch.from_numpy(observation).unsqueeze(0)))
                R = value.data
            else:
                R = torch.zeros(1, 1)
            values.append(Variable(R))
            R = Variable(R)
            gae = torch.zeros(1, 1)
            for i in reversed(range(len(rewards))):
                R = self._discount * R + rewards[i]
                advantage = R - values[i]
                value_loss += 0.5 * advantage.pow(2)
                # Generalized Advantage Estimataion
                delta_t = rewards[i] + self._discount * values[i + 1].data \
                    - values[i].data
                gae = gae * self._discount + delta_t
                policy_loss -= log_policies[i] * Variable(gae) \
                    + 0.01 * entropies[i]

            optimizer.zero_grad()
            total_loss = policy_loss + 0.5 * value_loss
            total_loss.backward()
            torch.nn.utils.clip_grad_norm(model.parameters(), 40)

            self._share_grads(model)
            optimizer.step()
        env.close()

    def _share_grads(self, model):
        for param, shared_param in zip(model.parameters(),
                                       self._shared_model.parameters()):
            if shared_param.grad is not None:
                return
            shared_param._grad = param.grad
