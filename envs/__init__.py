from gym.envs.registration import register
from envs.pong_env import PongSinglePlayerEnv
from envs.pong_env import PongDoublePlayerEnv

register(id='Pong-2p-v0', entry_point='envs:PongDoublePlayerEnv')

register(id='Pong-1p-v0', entry_point='envs:PongSinglePlayerEnv')
