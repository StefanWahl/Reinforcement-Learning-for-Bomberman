import pickle
import random
from collections import namedtuple, deque
from typing import List
import events as e
from datetime import datetime
import os
import numpy as np
import json

##########################################################################
# Stuff we load
from agent_code.agent_agent_2.Experience_Buffer import Experience_Buffer
from agent_code.agent_agent_2.agent_code.agent_agent.Gradient_Boosting_Model import GBM_Model
from agent_code.agent_agent_2.Hyperparameters import *
##########################################################################


def record_stats(self, events, reward):
    # save the statistics
    self.stat_track["steps_survived"][-1] += 1
    self.stat_track["total_reward_of_round"][-1] += reward

    if "COIN_COLLECTED" in events:
        self.stat_track["coins_collected"][-1] += 1
    if "INVALID_ACTION" in events:
        self.stat_track["Invalid_actions"][-1] += 1
    if "KILLED_OPPONENT" in events:
        self.stat_track["killed_opponent"][-1] += 1
    if "KILLED_SELF" in events:
        self.stat_track["killed_self"][-1] += 1
    if "CRATE_DESTROYED" in events:
        self.stat_track["crates_destroyed"][-1] += 1


def setup_training(self):
    """
    Initialise self for training purpose.

    This is called after `setup` in callbacks.py.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    self.decay = DECAY
    self.p_zero = P_ZERO
    
    #Define dictionary to store teh progress of the training
    self.stat_track = {
        "round": [0],
        "steps_survived": [0],
        "coins_collected": [0],
        "total_reward_of_round": [0],
        "killed_opponent": [0],
        "killed_self": [0],
        "Invalid_actions": [0],
        "crates_destroyed": [0],
    }

    # Score, Index and Date to label the models
    self.score = 0
    self.model_index = 0
    self.date = str(datetime.today()).split()[0]

    # Add an Experience Buffer
    self.buffer = Experience_Buffer(D=FEATURE_DIM, gamma=GAMMA, N_max=320000)

    # Create a new Model if there is no model
    if not os.path.isfile("my-saved-model.pt"):
        self.model = GBM_Model(MODEL_PARAMETERS)

        # Initial fit to make things work
        X = np.ones((6, 10, FEATURE_DIM))
        Y = np.ones((6, 10))
        self.model.train(X, Y)

    # open the existing model
    else:
        with open("my-saved-model.pt", "rb") as file:
            self.model = pickle.load(file)

    #Create a new model, if there is no existing model and store it
    if FIRST_SAVE:
        with open(f"{FOLDER}Hyperparameters.json", "w") as file:
            json.dump(HYPERPARAMETERS, file)
        with open(f"{FOLDER}my-saved-model_{self.model_index}_{self.date}.pt", "wb") as file:
            pickle.dump(self.model, file)
        with open(f"my-saved-model.pt", "wb") as file:
            pickle.dump(self.model, file)
        #Set the highscore to zero
        with open(f"{FOLDER}highscore.txt", "r+") as file:
            file.write(str(-999999) + "                         ")


def game_events_occurred(self, old_game_state: dict, self_action: str, new_game_state: dict, events: List[str]):
    """
    Called once per step to allow intermediate rewards based on game events.

    When this method is called, self.events will contain a list of all game
    events relevant to your agent that occurred during the previous step. Consult
    settings.py to see what events are tracked. You can hand out rewards to your
    agent based on these events and your knowledge of the (new) game state.

    This is *one* of the places where you could update your agent.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    :param old_game_state: The state that was passed to the last call of `act`.
    :param self_action: The action that you took.
    :param new_game_state: The state the agent is in now.
    :param events: The events that occurred when going from  `old_game_state` to `new_game_state`
    """
    self.logger.debug(f'Encountered game event(s) {", ".join(map(repr, events))} in step {new_game_state["step"]}')

    #Penalty for this time step
    events.append(TIME)

    #Droppped Bomb in the first step
    if new_game_state["step"] == 1 and e.BOMB_DROPPED in events:
        events.append(BOMB_IN_FIRST_STEP)

    #Unaimed bomb
    if e.BOMB_DROPPED in events:
        #Is there something that can be blown up?
        pos = new_game_state["self"][3]

        if pos[0] % 2 != 0: close_environmentX = new_game_state["field"][pos[0],max(0,pos[1]-3):min(16,pos[1]+3)]
        else:close_environmentX = np.array([])

        if  pos[1] % 2 != 0: close_environmentY = new_game_state["field"][max(0,pos[0]-3):min(16,pos[0]+3),pos[1]]
        else:close_environmentY =np.array([])

        if 1 not in close_environmentX and 1 not in close_environmentY: events.append(NOT_AIMED_BOMB)
        else: 
            aimed_targets = (close_environmentX == 1).sum() + (close_environmentY == 1).sum()
            for i in range(aimed_targets):
                events.append(AIMED_BOMB)

        #Is there an agent to blow up?
        for other in old_game_state["others"]:
            if pos[0] % 2 != 0 and other[3][0] == pos[0] and abs(other[3][0] - pos[0]) < 4: 
                events.append(AIMED_BOMB_OPPONENT)

            elif pos[1] % 2 != 0 and other[3][1] == pos[0] and abs(other[3][1] - pos[1]) < 4: 
                events.append(AIMED_BOMB_OPPONENT)

    if old_game_state != None:
        # Step 1: Get the reward
        reward = reward_from_events(self, events)
        self.score += reward

        # step 2: Transform the state into a feature vector
        X = self.state_to_features(old_game_state)

        # step 3: Store the transition
        self.buffer.record(X, ACTION_DICT[self_action], reward)

        #step 4: Save the stats
        record_stats(self,events,reward)
        self.stat_track["round"][-1] = old_game_state["round"]


def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):
    """
    Called at the end of each game or when the agent died to hand out final rewards.

    This is similar to reward_update. self.events will contain all events that
    occurred during your agent's final step.

    :param self: The same object that is passed to all of your callbacks.
    """
    self.logger.debug(f'Encountered event(s) {", ".join(map(repr, events))} in final step')

    # step 1: Get rewards
    reward = reward_from_events(self, events)
    self.score += reward

    # step 2: Transform the state into a feature vector
    X = self.state_to_features(last_game_state)

    # step 3: Store the transition
    self.buffer.record(X, ACTION_DICT[last_action], reward)

    # step 4: Update the buffer
    self.buffer.end_of_game()

    if last_game_state["round"] % TRAIN_FREQ == 0 and self.stat_track["round"][-1] > 1000:
        # step 5: Get Trainingsdate
        X, Y = self.buffer.get_data(BATCH_SIZE)

        # step 6: Update the model
        self.model.train(X, Y)

        # step 7: Store the model
        self.score = int(self.score / TRAIN_FREQ)
        self.model_index += 1
        highscore = None
        with open(f"{FOLDER}highscore.txt", "r+") as file:
            highscore = int(file.read())
            if self.score >= highscore:
                file.seek(0)
                file.write(str(self.score) + "                         ")

                with open(f"my-saved-model.pt", "wb") as file3:
                    pickle.dump(self.model, file3)
            
        with open(f"{FOLDER}my-saved-model_{self.model_index}_{self.score}_{self.date}.pt", "wb") as file2:
            pickle.dump(self.model, file2)
            
        # step 8: Print Score
        rand_ratio = np.round(self.rand_act / self.tot_act * 100, 1)
        print("Avg Score:", self.score, " Rand %:",
              rand_ratio, ", Index:", self.model_index)
        self.score = 0
        self.rand_act = 0
        self.tot_act = 0

    record_stats(self,events,reward)

    for k in self.stat_track.keys():
        self.stat_track[k].append(0)

    if last_game_state["round"] % SAVE_FREQ == 0:
        #Divide recorded rewarrd to get the mean reward of this round
        self.stat_track["total_reward_of_round"][0] /= self.stat_track["steps_survived"][0]

        #step 9: Save stats
        stats = list(self.stat_track.values())
        with open(FOLDER+"stats.txt","ab") as file:
            np.savetxt(file,np.array(stats).T)

        for k in self.stat_track.keys():
            self.stat_track[k] = [0]


def reward_from_events(self, events: List[str]):
    reward_sum = 0
    for event in events:
        if event in GAME_REWARDS:
            reward_sum += GAME_REWARDS[event]
    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    return reward_sum
