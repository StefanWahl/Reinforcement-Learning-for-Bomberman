import events as e
#Hyperparameters for Training
GAMMA = 0.8  # Discount factor for the MC value estimation
DECAY = 7500  # How fast the ratio of random actions decays. Set to approx 0.01 N with N being the number of rounds for training

P_ZERO = 0.1  # Starting value of ratio of random actions
FEATURE_DIM = 39  # Size of the feature vector
BATCH_SIZE = 100000  # Batch size for  training
N_MAX = 320000

TRAIN_FREQ = 100  # Number of rounds between training a new model
SAVE_FREQ = 400  # Number of rounds between saving data
FIRST_SAVE = False  # Save the model in setup? False if u want to keep a nice trained model but watch it
FOLDER = "test/"#"run_3_2021_03_21/"
#Commandline: python main.py play --agents agent_agent peaceful_agent rule_based_agent rule_based_agent --no-gui --n-rounds 40000 --train 1
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

# Mapping possible actions to numerical values
ACTION_DICT = {'UP': 0, 'RIGHT': 1, 'DOWN': 2, 'LEFT': 3, 'WAIT': 4, 'BOMB': 5}

# Rewards for Training
GAME_REWARDS = {
        e.COIN_COLLECTED: 1000,
        e.KILLED_OPPONENT: 5000,
        e.GOT_KILLED: -1500,
        e.KILLED_SELF: -2500,
        e.CRATE_DESTROYED:  +100,
        e.COIN_FOUND: +100,
        e.BOMB_DROPPED: 0,
        e.INVALID_ACTION: -100,
        e.WAITED: -2,
        BOMB_IN_FIRST_STEP:-100,
        NOT_AIMED_BOMB:-100,
        AIMED_BOMB:+100,
        AIMED_BOMB_OPPONENT: 500,
        MOVED_AWAY_FROM_BOMB:+100,
        TIME:-2,
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