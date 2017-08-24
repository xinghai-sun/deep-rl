import os
import argparse
import yaml
import gym
from gym import wrappers
import matplotlib.pyplot as plt
from agents.dqn_agent import DQNAgent
from agents.conv_dqn_agent import ConvDQNAgent
from agents.random_agent import RandomAgent
from agents.tabular_q_agent import TabularQAgent
from agents.a3c import A3CAgent
from envs.pong_env import Pong2PEnv
from wrappers.process_frame import RescaleFrameWrapper
from wrappers.process_frame import TransposeNormalizeWrapper


def create_async_agent(conf, env_generator):
    if conf['agent'] == "a3c":
        return A3CAgent(
            env_generator,
            learning_rate=conf['learning_rate'],
            discount=conf['discount'],
            t_max=conf['t_max'])
    else:
        raise ArgumentError("AsyncAgent type [%s] is not supported." %
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
        run_async(conf)


def run_async(conf):
    print("----- Running job [%s] ----- " % conf['job_name'])

    def env_generator():
        env = gym.make(conf['env'])
        env = RescaleFrameWrapper(env)
        env = TransposeNormalizeWrapper(env)
        return env

    agent = create_async_agent(conf, env_generator)
    agent.learn(conf['num_processes'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="conf/test_async.yaml",
        type=str,
        help="Configuration file in ymal format.")
    args = parser.parse_args()

    run_all()
