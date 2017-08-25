import argparse
import os
import yaml
import gym
from gym import wrappers
from agents.a3c import A3CAgent
from envs.pong_env import Pong1PEnv
from wrappers.process_frame import AtariRescale42x42Wrapper
from wrappers.process_frame import NormalizeWrapper


def create_async_agent(conf, action_space, observation_space):
    if conf['agent'] == "a3c":
        return A3CAgent(
            action_space,
            observation_space,
            learning_rate=conf['learning_rate'],
            discount=conf['discount'],
            t_max=conf['t_max'])
    else:
        raise ArgumentError("AsyncAgent type [%s] is not supported." %
                            conf['agent'])


def run_all():
    conf_list = yaml.load(file(args.config, 'r'))
    for conf in conf_list:
        run_async(conf)


def run_async(conf):
    print("----- Running job [%s] ----- " % conf['job_name'])

    def env_generator():
        env = gym.make(conf['env'])
        env = AtariRescale42x42Wrapper(env)
        env = NormalizeWrapper(env)
        return env

    env = env_generator()
    agent = create_async_agent(conf, env.action_space, env.observation_space)
    agent.learn_async(env_generator, conf['num_processes'], enable_test=True)
    env.close()


if __name__ == '__main__':
    os.environ['OMP_NUM_THREADS'] = '1'
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="conf/test_async.yaml",
        type=str,
        help="Configuration file in ymal format.")
    args = parser.parse_args()

    run_all()
