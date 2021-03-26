import os
import pickle
import random
import numpy as np
import sys

ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']

##########################################################################
# Stuff we load
from agent_code.agent_agent_1.feature_selection import features_V2 as get_features
from agent_code.agent_agent_1 import Gradient_Boosting_Model

def state_to_features(game_state: dict):
    return get_features(game_state)

#Handle changed directory layout
sys.modules["Gradient_Boosting_Model"] = Gradient_Boosting_Model
##############################################################################
def setup(self):
    """
    Setup your code. This is called once when loading each agent.
    Make sure that you prepare everything such that act(...) can be called.

    When in training mode, the separate `setup_training` in train.py is called
    after this method. This separation allows you to share your trained agent
    with other students, without revealing your training code.

    In this example, our model is a set of probabilities over actions
    that are is independent of the game state.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """

    # Case 1: No existing model and no training.
    if not os.path.isfile("my-saved-model.pt") and not self.train:
        self.logger.info("Setting up model from scratch.")
        raise EnvironmentError(
            "No existing model found. Please provide a model!")

    # case 2: Training
    elif self.train:
        self.state_to_features = state_to_features
        self.rand_act = 0
        self.tot_act = 0

    # Case 3: Evaluation mode: just load the trained model
    else:
        self.logger.info("Loading model from saved state.")
        with open("my-saved-model.pt", "rb") as file:
            self.model = pickle.load(file)


def act(self, game_state: dict):
    
    """
    Your agent should parse the input, think, and take a decision.
    When not in training mode, the maximum execution time for this method is 0.5s.

    :param self: The same object that is passed to all of your callbacks.
    :param game_state: The dictionary that describes everything on the board.
    :return: The action to take as a string.
    """

    # step 1: Transform the game state into features
    X = state_to_features(game_state)

    '''
    In Training one uses a epsilon greedy policy, in the evaluation mode, one uses the train policy
    '''
    # step 2: Get the presdiction
    #Training mode
    if self.train:
        # Implement epsilon greedy in the training file and load it into self
        Q = np.random.random()
        epsilon = float(self.decay / (self.decay + game_state["round"])) * self.p_zero
        if Q < epsilon:
            action_index = np.random.choice(np.arange(6), p = [1/6,1/6,1/6,1/6,1/6,1/6])
            self.rand_act += 1
            self.tot_act += 1
        else:
            p = self.model(X)
            action_index = np.argmax(p[:6])
            self.tot_act += 1

    #Evaluation mode
    else:
        p = self.model(X)
        action_index = np.argmax(p[:6])

    return ACTIONS[action_index]
