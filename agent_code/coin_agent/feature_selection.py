import numpy as np

def coins_simple(game_state: dict) -> np.array:
    loc = game_state["self"][3]
    state = np.zeros((2, 2))

    #Step 1: Add the absolute position of the agent to feature vector
    state[0] = np.array(loc)
    min_dist = 100

    #Step 2: Get the relative coordinates of the closest coin and add it to the feature vector
    for coin in game_state["coins"]:
        dist = abs(coin[0] - loc[0]) + abs(coin[1] - loc[1])
        if dist < min_dist:
            state[1] = np.array([coin[0] - loc[0], coin[1] - loc[1]])

    #Step 3: Return the feature vector
    return state.flatten()