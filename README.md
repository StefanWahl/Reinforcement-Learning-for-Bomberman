# Reinforcement-Learning-for-Bomberman

This repository contains our results for the final project for the Lecture "Fundamentals of Machine Learning", winter term 2020/2021 at the university of Heidelberg.

The task of this projct was, to use reinforcement learning to train an agent which can successly play the game Bomberman. The general framework of the game is taken from https://github.com/ukoethe/bomberman_rl.

## data
The data sets recorded during the training can be found in "data". The colums of the file stats are as follows:

* Column 1: Training round
* Column 2: Number of steps the agent survived in the current round
* Column 3: Number of coins collected during the training
* Column 4: Total auxillary reward collected during the round
* Column 5: Number of opponents killed in the round
* Column 6: 1 if the agent killed it self, else 0
* Column 7: Number of invalid actions in the current round
* Column 8: Number of crates destroyed in the current round
