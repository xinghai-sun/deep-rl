import os
import argparse
import yaml
import gym
from gym import wrappers
#import matplotlib.pyplot as plt
from agents.dqn_agent import DQNAgent
from agents.conv_dqn_agent import ConvDQNAgent
from agents.random_agent import RandomAgent
from agents.tabular_q_agent import TabularQAgent
from envs.pong_env import PongSinglePlayerEnv
from wrappers.process_frame import AtariRescale42x42Wrapper
from wrappers.process_frame import NormalizeWrapper


def create_env(conf):
    env = gym.make(conf['env'])
    env.seed(0)
    return env


def wrapper_env(conf, env):
    if conf['monitor_dir']:
        env = wrappers.Monitor(env, conf['monitor_dir'], force=True)
    if conf['agent'] == "conv_dqn":
        env = AtariRescale42x42Wrapper(env)
        env = NormalizeWrapper(env)
    return env


def create_agent(conf, action_space, observation_space):
    if conf['agent'] == "dqn":
        return DQNAgent(
            action_space,
            observation_space,
            batch_size=conf['batch_size'],
            learning_rate=conf['learning_rate'],
            discount=conf['discount'],
            epsilon=conf['random_explore'])
    elif conf['agent'] == "conv_dqn":
        return ConvDQNAgent(
            action_space,
            observation_space,
            batch_size=conf['batch_size'],
            learning_rate=conf['learning_rate'],
            discount=conf['discount'],
            epsilon=conf['random_explore'])
    elif conf['agent'] == "tabular_q":
        return TabularQAgent(
            action_space,
            observation_space,
            q_init=conf['q_value_init'],
            learning_rate=conf['learning_rate'],
            discount=conf['discount'],
            epsilon=conf['random_explore'])
    elif conf['agent'] == "random":
        return RandomAgent(action_space, observation_space)
    else:
        raise ArgumentError("Agent type [%s] is not supported." %
                            conf['agent'])


def plot_return(return_list, filepath):
    plt.plot(return_list)
    plt.xlabel('Episode')
    plt.ylabel('Cummulative Rewards Per Episode')
    plt.grid()
    dirname = os.path.dirname(filepath)
    if not os.path.exists(dirname):
        os.mkdir(dirname)
    plt.savefig(filepath)


def run_all():
    conf_list = yaml.load(file(args.config, 'r'))
    for conf in conf_list:
        return_list = run(conf)
        plot_return(return_list, os.path.join('figures', conf['job_name']))


def run(conf):
    print("----- Running job [%s] ----- " % conf['job_name'])
    env = create_env(conf)
    env = wrapper_env(conf, env)
    agent = create_agent(conf, env.action_space, env.observation_space)
    return_list = []
    for episode in xrange(conf['num_episodes']):
        cum_return = 0.0
        observation = env.reset()
        done = False
        while not done:
            #env.render()
            action = agent.act(observation)
            next_observation, reward, done, _ = env.step(action)
            agent.learn(reward, next_observation, done)
            observation = next_observation
            cum_return += reward
        return_list.append(cum_return)
        print("Episode %d/%d Return: %f." %
              (episode + 1, conf['num_episodes'], cum_return))
    env.close()
    return return_list


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="conf/example.yaml",
        type=str,
        help="Configuration file in ymal format.")
    args = parser.parse_args()

    run_all()
