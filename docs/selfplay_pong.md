# Self-play Learning for Pong Game

## Introduction

- A newly developed (with pygame) [**PongGame** gym environment](https://github.com/xinghai-sun/deep-rl/blob/master/envs/pong_env.py), supporting both **single player** mode (battle with built-in player) and **double player** mode (battle with itself, or any other external player). Notice that the original Pong-v2 open-ai gym env does not support two-player mode.
- An [**A3C** Agent](https://github.com/xinghai-sun/deep-rl/blob/master/agents/a3c.py) implementation with PyTorch. [Other RL](https://github.com/xinghai-sun/deep-rl/tree/master/agents) algorithms (e.g. DQN) are also provided.
- **Self-play** learning settings: for each episode, we randomly select an opponent player from some recent copies of the learning agent itself. Let the learning agent play with this opponent, learn from experiences, and become stronger. Since the opponent (a random history copy of itself) also becomes stronger than before, it forces the learning agent to grow even stronger.


## Experiments

We compare three players:

- **Built-in**: rule-based built-in player.
- **A3C-basic**: A3C agent learning from combating with the Built-in player. It only experiences $360 \times 16$ episodes with 16 asynchronous processes, due to the limited time.
- **A3C-selfplay**: A3C agent learning from combating with itself (its history version). It only experiences $360 \times 16$ episodes with 16 asynchronous processes, due to the limited time.

<div align="center">
<img src="image/single_vs_auto_360.gif"></br>
Fig 1. A3C-basic vs. Built-in
</div>

<div align="center">
<img src="image/selfplay_vs_auto.gif"></br>
Fig 2. A3C-selfplay vs. Built-in
</div>


<div align="center">
<img src="image/selfplay_vs_single.gif"></br>
Fig 3. A3C-selfplay vs. A3C-basic
</div>

### Learning Speed

<div align="center">
<img src="image/learning_curve.png", width=400></br>
Fig 4. Learning curves of A3C-selfplay agents (play with itself) and A3C-basic agents (play with the built-in rule-based player). The figure plots the average (over 10 random episodes) cumulative rewards of  evaluative battle results combatting with the built-in player as the opponent.
</div>

### Scoreboard

Player VS. Player | Average Score (20 games)
:-----------: | :------------:
A3C-Basic VS. Built-in  | 18.5 : 1.5
A3C-Selfplay VS. Built-in  | 9.6 : 10.4
A3C-Selfplay VS. A3C-Basic | 14.5 : 5.5

Notice that even the A3C-selfplay agent did not play as perfect as the A3C-basic agent when combating with the built-in player. However, when combating with each other, A3C-selfplay outperforms A3C-basic, overwhelmingly and interestingly.



## Quick Start

### Preliminaries

1. Install pytorch 0.1.10.post2
2. Install dependencies:

```
pip install -r requirements.txt
```

### Run A3C-basic

```
python run_async.py --config conf/a3c_my-pong.yaml
```

### Run A3C-selfplay

```
python run_selfplay.py --config conf/a3c_my-selfplay_my-pong.yaml
```

### Run Evaluation
```
python eval.py --mode combat --config conf/a3c_selfplay_my-pong.yaml --eval_model_path checkpoints/agent.model-360 --eval_opponent_path checkpoints/agent.model-100
```
