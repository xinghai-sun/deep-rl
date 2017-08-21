from gym.envs.registration import register
from envs.pong_env import Pong1PEnv
from envs.pong_env import Pong2PEnv

register(
    id='Pong-2p-v0',
    entry_point='envs:Pong2PEnv', )

register(
    id='Pong-1p-v0',
    entry_point='envs:Pong1PEnv', )
