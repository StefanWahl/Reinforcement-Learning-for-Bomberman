import events as e
#Hyperparameters for Training
GAMMA = 0.8  # Discount factor for the MC value estimation
DECAY = 3000  # How fast the ratio of random actions decays. Set to approx 0.01 N with N being the number of rounds for training

P_ZERO = 1  # Starting value of ratio of random actions

FEATURE_DIM = 39  # Size of the feature vector
BATCH_SIZE = 15000  # Batch size for  training
TRAIN_FREQ = 100  # Number of rounds between training a new model
SAVE_FREQ = 200  # Number of rounds between saving data
FIRST_SAVE = True  # Save the model in setup? False if u want to keep a nice trained model but watch it
FOLDER = "run_2_2021_03_25/"

MODEL_PARAMETERS = {
    "learning_rate": 0.1, 
    "n_estimators": 300, 
    "max_depth": 6, 
    "subsample": 0.95, 
    "colsample_bytree": 1
    }  # Parameters for the model

# Our Events
TIME = "TIME"
BOMB_IN_FIRST_STEP = "BOMB_IN_FIRST_STEP"
NOT_AIMED_BOMB = "NOT_AIMED_BOMB"
MOVED_AWAY_FROM_BOMB = "MOVED_AWAY_FROM_BOMB"
AIMED_BOMB = "AIMED_BOMB"
AIMED_BOMB_OPPONENT = "AIMED_BOMB_OPPONENT"
N_MAX = 120000

# Mapping possible actions to numerical values
ACTION_DICT = {'UP': 0, 'RIGHT': 1, 'DOWN': 2, 'LEFT': 3, 'WAIT': 4, 'BOMB': 5}

# Rewards for Training
GAME_REWARDS= {
    "COIN_COLLECTED": 200, 
    "KILLED_SELF": -1000, 
    "CRATE_DESTROYED": 200, 
    "COIN_FOUND": 100, 
    "NOT_AIMED_BOMB": -100, 
    "AIMED_BOMB": 100, 
    "MOVED_AWAY_FROM_BOMB": 100, 
    "TIME": -2
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