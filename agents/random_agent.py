from agents.base_agent import BaseAgent


class RandomAgent(BaseAgent):
    '''Random agent.'''

    def __init__(self, action_space, observation_space):
        self._action_space = action_space

    def act(self, observation, greedy=False):
        return self._action_space.sample()

    def learn(self, reward, next_observation, done):
        pass
