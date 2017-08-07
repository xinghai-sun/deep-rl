import argparse
import gym
from agents.dqn_agent import DQNAgent

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--env_id",
    default="CartPole-v0",
    type=str,
    help="Select the environment to run.")
parser.add_argument(
    "--num_episodes", default=2000, type=int, help="Episode number to run.")
parser.add_argument(
    "--monitor_output",
    default="",
    type=str,
    help="Path to output monitor results.")
args = parser.parse_args()


def run():
    # prepare env
    env = gym.make(args.env_id)
    env.seed(0)
    if args.monitor_output:
        env = wrappers.Monitor(env, args.monitor_output, force=True)
    agent = DQNAgent(env.action_space, env.observation_space)
    # simulate
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
        print("Return: %f." % cum_return)
    # clean up
    env.close()


if __name__ == '__main__':
    run()
