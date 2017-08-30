import argparse
import os
import yaml
import copy
import gym
from gym import wrappers
from agents.a3c import A3CAgent
import torch.multiprocessing as mp
from envs.pong_env import PongSinglePlayerEnv
from wrappers.process_frame import AtariRescale42x42Wrapper
from wrappers.process_frame import NormalizeWrapper


def create_agent(conf, action_space, observation_space):
    if conf['agent'] == "a3c":
        return A3CAgent(
            action_space,
            observation_space,
            learning_rate=conf['learning_rate'],
            discount=conf['discount'],
            t_max=conf['t_max'])
    else:
        raise ArgumentError("Agent type [%s] is not supported." %
                            conf['agent'])


def create_env(conf, monitor_on=False):
    env = gym.make(conf['env'])
    if conf['monitor_dir'] != '' and monitor_on:
        env = wrappers.Monitor(env, conf['monitor_dir'], force=True)
    if conf['use_atari_wrapper']:
        env = AtariRescale42x42Wrapper(env)
        env = NormalizeWrapper(env)
    return env


def run_eval(conf, num_episodes, eval_model_path):
    env = create_env(conf, monitor_on=True)

    agent = create_agent(conf, env.action_space, env.observation_space)
    agent.load_model(eval_model_path)

    sum_return = 0.0
    for episode in xrange(num_episodes):
        cum_return = 0.0
        observation = env.reset()
        done = False
        while not done:
            action = agent.act(observation, greedy=True)
            next_observation, reward, done, _ = env.step(action)
            observation = next_observation
            cum_return += reward
        sum_return += cum_return
        print("Episode %d/%d Return: %f." %
              (episode + 1, num_episodes, cum_return))
    print("Average Return: %f." % (sum_return / num_episodes))
    env.close()


def run_selfplay_eval(conf, num_episodes, eval_model_path):
    env = create_env(conf, monitor_on=True)

    agent = create_agent(conf, env.action_space.spaces[0],
                         env.observation_space.spaces[0])
    agent.load_model(eval_model_path)
    opponent_agent = copy.deepcopy(agent)

    sum_return = 0.0
    for episode in xrange(num_episodes):
        cum_return = 0.0
        observation = env.reset()
        done = False
        agent.reset()
        opponent_agent.reset()
        while not done:
            obs_A, obs_B = observation
            action_A = agent.act(obs_A, greedy=True)
            action_B = opponent_agent.act(obs_B, greedy=True)
            next_observation, reward, done, _ = env.step((action_A, action_B))
            observation = next_observation
            cum_return += reward[0]
        sum_return += cum_return
        print("Episode %d/%d Return: %f." %
              (episode + 1, num_episodes, cum_return))
    print("Average Return: %f." % (sum_return / num_episodes))
    env.close()


def run_combat_eval(conf, num_episodes, eval_model_path, eval_opponent_path):
    env = create_env(conf, monitor_on=True)

    agent = create_agent(conf, env.action_space.spaces[0],
                         env.observation_space.spaces[0])
    agent.load_model(eval_model_path)

    opponent_agent = create_agent(conf, env.action_space.spaces[0],
                                  env.observation_space.spaces[0])
    opponent_agent.load_model(eval_opponent_path)

    sum_return = 0
    for episode in xrange(num_episodes):
        cum_return = 0.0
        observation = env.reset()
        done = False
        agent.reset()
        opponent_agent.reset()
        while not done:
            obs_A, obs_B = observation
            action_A = agent.act(obs_A, greedy=True)
            action_B = opponent_agent.act(obs_B, greedy=True)
            next_observation, reward, done, _ = env.step((action_A, action_B))
            observation = next_observation
            cum_return += reward[0]
        sum_return += cum_return
        print("Episode %d/%d Return: %f." %
              (episode + 1, num_episodes, cum_return))
    print("Average Return: %f." % (sum_return / num_episodes))
    env.close()


if __name__ == '__main__':
    os.environ['OMP_NUM_THREADS'] = '1'
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Configuration file in ymal format.")
    parser.add_argument(
        "--num_episodes",
        type=int,
        default=10,
        help="Number of evaluative episodes.")
    parser.add_argument(
        "--mode",
        type=str,
        default='single',
        choices=['single', 'selfplay', 'combat'],
        help="Play mode: single, selfplay, combat.")
    parser.add_argument(
        "--eval_model_path",
        type=str,
        required=True,
        help="Model path to evaluate.")
    parser.add_argument(
        "--eval_opponent_path",
        type=str,
        default=None,
        help="Model path for the opponent agent, only used in combat mode.")
    args = parser.parse_args()

    conf = yaml.load(file(args.config, 'r'))

    if args.mode == 'selfplay':
        run_selfplay_eval(conf, args.num_episodes, args.eval_model_path)
    elif args.mode == 'single':
        run_eval(conf, args.num_episodes, args.eval_model_path)
    elif args.mode == 'combat':
        run_combat_eval(conf, args.num_episodes, args.eval_model_path,
                        args.eval_opponent_path)
