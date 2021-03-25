import events as e
#Hyperparameters for Training
GAMMA = 0.8  # Discount factor for the MC value estimation
DECAY = 2000  # How fast the ratio of random actions decays. Set to approx 0.01 N with N being the number of rounds for training
P_ZERO = 0.25  # Starting value of ratio of random actions
FEATURE_DIM = 39  # Size of the feature vector
BATCH_SIZE = 25000  # Batch size for  training
TRAIN_FREQ = 100  # Number of rounds between training a new model
SAVE_FREQ = 400  # Number of rounds between saving data
FIRST_SAVE = False  # Save the model in setup? False if u want to keep a nice trained model but watch it
FOLDER = "run_2_2021_03_16/"
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

# Mapping possible actions to numerical values
ACTION_DICT = {'UP': 0, 'RIGHT': 1, 'DOWN': 2, 'LEFT': 3, 'WAIT': 4, 'BOMB': 5}


'''
Idee Rewardshaping für die Kisten:
    -Bombenlegen sanktionierte mit dem gleichen wert der beim Zerszören von zwei kisten gewonnen wird. 
     Soll inflationäres Bomben legen verhindern
    -Abzug wenn er im ersten Schritt eine Bombe legt
    -Abzug wenn keine Kiste näher als zwei Felder ist und eine Bombe gelegt wird soll ungezielte Bomben verhindern
    -Belohnung wenn er Abstand zu gesetzter Bombe erhöht

'''
# Rewards for Training
GAME_REWARDS = {
        e.COIN_COLLECTED: 600,
        e.KILLED_SELF: -1000,
        e.CRATE_DESTROYED:  +100,
        e.COIN_FOUND: +100,
        e.BOMB_DROPPED: 0,
        e.INVALID_ACTION: -100,
        e.WAITED: -2,
        BOMB_IN_FIRST_STEP:-100,
        NOT_AIMED_BOMB:-100,
        AIMED_BOMB:+100,
        MOVED_AWAY_FROM_BOMB:+100,
        TIME:-2,
        e.WAITED:-2
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