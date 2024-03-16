# MFG-RL-PIDL

We formulate the Spatiotemporal Mean Field Games to value iteration expression and use DDPG and residual networks to solve the system dynamics. We apply this method to the traffic ringroad with 3 different reward functions, obtaining the following results: 

![result](./results.png)

Our conference paper "A Hybrid Framework of Reinforcement Learning and Physics-Informed Deep Learning for Spatiotemporal Mean Field Games" has been published in AAMAS 2023. If you find it helpful, please cite it.

```
@inproceedings{10.5555/3545946.3598748,
author = {Chen, Xu and Liu, Shuo and Di, Xuan},
title = {A Hybrid Framework of Reinforcement Learning and Physics-Informed Deep Learning for Spatiotemporal Mean Field Games},
year = {2023},
isbn = {9781450394321},
publisher = {International Foundation for Autonomous Agents and Multiagent Systems},
address = {Richland, SC},
booktitle = {Proceedings of the 2023 International Conference on Autonomous Agents and Multiagent Systems},
pages = {1079–1087},
numpages = {9},
keywords = {mean field games, physics-informed deep learning, reinforcement learning},
location = {London, United Kingdom},
series = {AAMAS '23}
}
```

## How to Run

```
python MFG.py # hyperparams set by default
```
