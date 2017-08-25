from gym.envs.registration import register
from envs.pong_env import Pong1PEnv
from envs.pong_env import Pong2PEnv

register(
    id='Pong-2p-v0',
    entry_point='envs:Pong2PEnv',
    kwargs={
        'screen_size': (400, 400),
        'bat_height': 50,
        'bat_speed': 2,
        'ball_speed': 2,
        'max_round': 20
    })

register(
    id='Pong-1p-v0',
    entry_point='envs:Pong1PEnv',
    kwargs={
        'screen_size': (400, 400),
        'bat_height': 100,
        'bat_speed': 5,
        'ball_speed': 8,
        'max_round': 3
    })
