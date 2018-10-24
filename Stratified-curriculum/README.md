# Stratified Curriculum for DoorKey

Applying stratified curriculum for DoorKey environment.

For this task some changes had to be done in [MiniGrid](https://github.com/maximecb/gym-minigrid) and [RL library](https://github.com/lcswillems/pytorch-a2c-ppo) repositories.

Changes can be made by changing some code (just copy files from 'Changes' folder and replace them in needed locations).

To start the test a script 'stratified_curriculum-starter.py' was made. it calls training script in rl-torch repository (with changes made).

# Changes made to MiniGrid

Changes are made wrt gym-minigrid/minigrid.py and gym-minigrid/ym_minigrid/doorkey.py files. Just replace those files.

# Changes made to rl-torch

Changes are made wrt scripts/train.py file. Just replace those files.
