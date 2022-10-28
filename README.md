# MFG-RL-PIDL

This is the source code for paper - A Hybrid Framework of Reinforcement Learning and Physics-Informed Deep Learning for Spatiotemporal Mean Field Games.

## File Structure

- `data` folder includes the numerical results of MFGs.
- `MFG-RL-PIDL.py` is the runner of Alg. 1 MFG RL-PIDL in our paper.
- `value_iteration_DDPG.py` is the training function of Alg. 1 MFG RL-PIDL.
- `MFG-Pure-PIDL-*.ipynb` are the source codes of the Alg. 2 MFG-Pure-PIDL.
- `model.py` includes the PyTorch network models of $\rho$ -Net, V-Net and u-net.
- `uitls.py` are the implementations of auxiliary function.