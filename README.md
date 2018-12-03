# MinGrid-Improved-RL-Methods
Applying curiosity and curriculum to [MinGrid](https://github.com/maximecb/gym-minigrid) from gym using an RL [implementation](https://github.com/lcswillems/pytorch-a2c-ppo) with PPO.

In the purpose of learning AI, this repository will contain all the results of my tests.
The idea is to use MinGrid as a problem and use a working RL implementation to train our agent.

# Methods:
We start by studying the following papers:
- [Reverse Curriculum Generation for Reinforcement Learning](http://proceedings.mlr.press/v78/florensa17a/florensa17a.pdf)
- [Accuracy-based Curriculum Learning in Deep Reinforcement Learning](https://arxiv.org/pdf/1806.09614.pdf)
- [Curiosity-driven Exploration by Self-supervised Prediction](https://pathak22.github.io/noreward-rl/resources/icml17.pdf)

Once gained full knowledge we'll write some python code to run our tasks for us.
Then view results with tensorboard, take some snapshots and write some paper for ourself!

# Analysis
Inside each results-folder you can find various screenshots and an info.txt file where i put my conclusions.

# Folders
Repository have folling folders:
-All-results-1e6-frames: contains basic curriculum training from 5x5, 6x6, 8x8, 16x16 with  1e6 frames
-All-results-5-10e6: contains basic curriculum training from 5x5, 6x6, 8x8, 16x16 with 5x1e6 frames
-Curriculum: contains real results using basic curriculum (using 5x5, 6x6, 8x8, 10x10, 12x12, 14x14, 16x16) with graphs and descriptions
-Stratified-curriculum: contains working thesis data.
  * Interesting results are in folder __Results/16x16__.
  * __Changes__ folder contain all files changed in Gym-MiniGrid and rl-torch (just clone basic repositories and paste our changes, it will work).
  * __stratified_curriculum-starter.py__ is a script to train with stratified curriculum. it have to be placed inside rl-torch folder (because it calls scripts/train.py script).
  * __est-stratified-positions-generations.py__ is a script to test each stratified_int generator for integers based on delta_strat and sigma
