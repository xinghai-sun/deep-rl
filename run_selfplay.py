import argparse
import os
import copy
import yaml
import gym
import random
from collections import deque
from gym import wrappers
from agents.a3c import A3CAgent
import torch.multiprocessing as mp
from envs.pong_env import PongSinglePlayerEnv
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


def run_async(conf):
    print("----- Running job [%s] ----- " % conf['job_name'])

    def create_env(monitor_on=False):
        env = gym.make(conf['env'])
        if conf['monitor_dir'] != '' and monitor_on:
            env = wrappers.Monitor(env, conf['monitor_dir'], force=True)
        if conf['use_atari_wrapper']:
            env = AtariRescale42x42Wrapper(env)
            env = NormalizeWrapper(env)
        return env

    env = create_env()
    master_agent = create_async_agent(conf, env.action_space.spaces[0],
                                      env.observation_space.spaces[0])
    env.close()
    agent_team = deque(maxlen=20)
    agent_team.append(copy.deepcopy(master_agent))

    def learn_thread(process_id):
        env = create_env(monitor_on=process_id == 0)
        env.seed(process_id)
        slave_agent = master_agent.create_async_learner()
        return_list = []
        for episode in xrange(conf['num_episodes_per_process']):
            cum_return = 0.0
            observation = env.reset()
            done = False
            opponent_agent = random.choice(agent_team)
            while not done:
                obs_A, obs_B = observation
                action = slave_agent.act(obs_A)
                opponent_action = opponent_agent.act(obs_B)
                next_observation, reward, done, _ = env.step(
                    (action, opponent_action))
                next_obs_A, next_obs_B = next_observation
                reward_A, reward_B = reward
                slave_agent.learn(reward_A, next_obs_A, done)
                observation = next_observation
                cum_return += reward_A
            return_list.append(cum_return)
            print("Episode %d/%d Return: %f." %
                  (episode + 1, conf['num_episodes_per_process'], cum_return))
            agent_team.append(copy.deepcopy(master_agent))
        env.close()

    processes = []
    for process_id in range(0, conf['num_processes']):
        p = mp.Process(target=learn_thread, args=(process_id, ))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()


if __name__ == '__main__':
    os.environ['OMP_NUM_THREADS'] = '1'
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Configuration file in ymal format.")
    args = parser.parse_args()

    conf = yaml.load(file(args.config, 'r'))
    run_async(conf)
