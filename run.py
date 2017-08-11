import argparse
import gym
import matplotlib.pyplot as plt
from agents.dqn_agent import DQNAgent
from agents.random_agent import RandomAgent
from agents.tabular_q_agent import TabularQAgent

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--env_id",
    default="CartPole-v0",
    type=str,
    help="Select the environment to run.")
parser.add_argument(
    "--num_episodes", default=500, type=int, help="Episode number to run.")
parser.add_argument(
    "--monitor_output",
    default="",
    type=str,
    help="Path to output monitor results.")
parser.add_argument(
    "--agent_type",
    default="dqn",
    type=str,
    help="Agent type name. Options: dqn, random, tabular_q")
args = parser.parse_args()


def create_agent(agent_type, action_space, observation_space, **kwargs):
    if agent_type == "dqn":
        return DQNAgent(action_space, observation_space, **kwargs)
    elif agent_type == "random":
        return RandomAgent(action_space, observation_space, **kwargs)
    elif agent_type == "tabular_q":
        return TabularQAgent(action_space, observation_space, **kwargs)
    else:
        raise ArgumentError("Agent type [%s] is not supported." % agent_type)


def plot(return_list):
    plt.plot(return_list)
    plt.show()


def run():
    # prepare env
    env = gym.make(args.env_id)
    env.seed(0)
    if args.monitor_output:
        env = wrappers.Monitor(env, args.monitor_output, force=True)
    # prepare agent
    agent = create_agent(args.agent_type, env.action_space,
                         env.observation_space)
    # simulate
    return_list = []
    for episode in xrange(args.num_episodes):
        cum_return = 0.0
        observation = env.reset()
        done = False
        while not done:
            action = agent.act(observation)
            next_observation, reward, done, _ = env.step(action)
            agent.learn(observation, action, reward, next_observation, done)
            observation = next_observation
            cum_return += reward
        return_list.append(cum_return)
        print("Return: %f." % cum_return)
    plot(return_list)
    # clean up
    env.close()


if __name__ == '__main__':
    run()
