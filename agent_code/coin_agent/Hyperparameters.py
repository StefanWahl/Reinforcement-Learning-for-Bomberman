import events as e

#Hyperparameters for Training
GAMMA = 0.9  # Discount factor for the MC value estimation
DECAY = 1616  # How fast the ratio of random actions decays. Set to approx 0.01 N with N being the number of rounds for training
P_ZERO = 1  # Starting value of ratio of random actions
FEATURE_DIM = 4  # Size of the feature vector
BATCH_SIZE = 10000  # Batch size for  training
N_MAX = 120000#Buffer size
TRAIN_FREQ = 100  # Number of rounds between training a new model
SAVE_FREQ = 200  # Number of rounds between saving data
FIRST_SAVE = True  # Save the model in setup? False if u want to keep a nice trained model but watch it
FOLDER = "run_1/"

MODEL_PARAMETERS = {
    "learning_rate": 0.1,
    "n_estimators": 300,
    "max_depth": 3
}  # Parameters for the model

# Our Events
TIME = "TIME"

# Mapping possible actions to numerical values
ACTION_DICT = {'UP': 0, 'RIGHT': 1, 'DOWN': 2, 'LEFT': 3, 'WAIT': 4, 'BOMB': 5}

# Rewards for Training
GAME_REWARDS = {
        e.COIN_COLLECTED: 200,
        TIME: -1,
        e.KILLED_SELF: -1000
}

# Dict containing all relevant Hyperparameters
HYPERPARAMETERS = {
    "GAMMA": GAMMA,
    "DECAY": DECAY,
    "P_ZERO": P_ZERO,
    "FEATURE_DIM": FEATURE_DIM,
    "BATCH_SIZE": BATCH_SIZE,
    "TRAIN_FREQ": TRAIN_FREQ,
    "SAVE_FREQ": SAVE_FREQ,
    "FIRST_SAVE": FIRST_SAVE,
    "MODEL_PARAMETERS": MODEL_PARAMETERS,
    "GAME_REWARDS": GAME_REWARDS
}