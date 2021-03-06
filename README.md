# Reinforcement-Learning-for-Bomberman
## By Leon Marx, Johannes Schmidt and Stefan Wahl

This repository contains our results for the final project for the Lecture "Fundamentals of Machine Learning", winter term 2020/2021 at the university of Heidelberg.

The task of this projct was, to use reinforcement learning to train an agent which can successly play the game Bomberman. The general framework of the game is taken from https://github.com/ukoethe/bomberman_rl.

## Required modules

For our implementations we used Python 3.8.. To run the code the following modules exceeding the python 3 standart library are needed:

* xgboost (version we used: V1.3.3)
* pygame (version we used: V2.0.1)
* tqdm (version we used: V4.47.0)

## Agent code

The folder agent_code contains the implementataion for a rule based agent, a random agent and a peaceful agent as they were provided by the framework. Besides these three agents, this folder contains the agents we trained:

* coin_agent and coin_agent_2: To run this agent, the value of CRATE_DENSITY has to be set to 0.0 and MAX_AGENTS has to be set to 1 in the file settings.py
* crate_agent: To run this the number of agents has to be reduced (MAX_AGENTS = 1 in file settings.py)
* agent_agent_1, agent_agent_2, agent_agent_3

Details about the different agents can be found in our report. Ech of the agent folders contains all the code used to train the agent and to run the agent. The file Hyperparameters summarizes the hyperparameters used to train the agent. The class definitions of the experiende buffer in Experience_Buffer.py and the definition of our model in Gradient_Boossting_Model.py are equivalent for all the trained agents. The feature selection in feature_selection.py is similar for all the agents exept coin_agent. Hyperparameters.py and callbacks.py are differnt for most of the agents due to due to different tasks and different training. train.py is similar for all the agents, there are minor changes for the different agents.

## data
The data sets recorded during the training can be found in "data". The colums of the file "stats.txt" can be read as follows:

* Column 1: Training round
* Column 2: Number of steps the agent survived in the current round
* Column 3: Number of coins collected during the training
* Column 4: Total auxillary reward collected during the round
* Column 5: Number of opponents killed in the round
* Column 6: 1 if the agent killed it self, else 0
* Column 7: Number of invalid actions in the current round
* Column 8: Number of crates destroyed in the current round

For the agent_agents, there is also a file called "performance_test.txt". This file contains the score of the different models trained in this run over 50 games without any random acitons.
