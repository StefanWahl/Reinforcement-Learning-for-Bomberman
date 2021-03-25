import numpy as np

'''
This class is meant to store the transitions during the training.
'''
class Experience_Buffer():
    def __init__(self,D,gamma,N_max = 20000):
        '''
        Parameters:
            gamma:      Discount factor
            N_max:      Maximal number of transitions stored for a single action        default: 800
            D:          Size of a feature vector

        '''
        #Store the discount factor
        self.gamma = gamma

        #Store the maximum lenght of the buffer
        self.N_max = N_max
        
        #Buffer to store the transitions with the rewards
        self.buffer = [{"states":np.zeros((N_max,D)),"responses":np.zeros(N_max),"size":0} for i in range(6)]

        #Temporary buffer to store the transitions of the current round
        self.temp_buffer = {"states":[],"rewards":[],"actions":[]}

    def record(self,features,action,reward):
        '''
        This functin can be called after a transition is made. The given transition is added to the temporary buffer.

        Parameters:
            features:   Current set of features
            action:     Current action
            rewad:      Reward for the action in this state
        '''
        self.temp_buffer["states"].append(features)
        self.temp_buffer["rewards"].append(reward)
        self.temp_buffer["actions"].append(action)

    def end_of_game(self):
        '''
            call this after the last round of the game is played to compute the responses by MC value estimation and transfer the content 
            form the temporary buffer to the buffer
        '''
        k = len(self.temp_buffer["states"])

        temp_rewards = np.array(self.temp_buffer["rewards"])
        temp_states =  np.array(self.temp_buffer["states"])
        temp_actions = np.array(self.temp_buffer["actions"])

        #Get the responses:
        respons = np.zeros(k)
        respons[-1] = temp_rewards[-1]
        for i in range(k-2,-1,-1):
            respons[i] = temp_rewards[i]+respons[i+1] * self.gamma

        #Updatet the buffer
        for i in range(6):
            mask = (temp_actions == i)
            t = np.sum(mask)
            s = self.buffer[i]["size"]
            if t == 0:
                continue

            #Case 1: the buffer has enough space for the new instances
            if s + t <= self.N_max:
                self.buffer[i]["responses"][s:s+t] = respons[mask]
                self.buffer[i]["states"][s:s+t] = temp_states[mask]
                self.buffer[i]["size"] += t
            
            #Case 2: the buffer has not enough space for the new instances
            else:
                self.buffer[i]["responses"] = np.concatenate((self.buffer[i]["responses"][t - self.N_max + s:],respons[mask]),axis = 0)
                self.buffer[i]["states"] = np.concatenate((self.buffer[i]["states"][t - self.N_max + s:],temp_states[mask]),axis = 0)
                self.buffer[i]["size"] = self.N_max

        #Reset the temporary buffer
        self.temp_buffer = {"states":[],"rewards":[],"actions":[]}

    def get_data(self,N):
        '''
        This function can be used, to get trainingsdate from the buffer.

        Parameters:
            N: Number of instances per action.

        Returns:
            X: Lsit containing six arrays with the states, one for each action  
            Y: List containing six arrays with the responses, one for each action
        '''
        X = []
        Y = []

        for i in range(6):
            #Avoid index out of range errors
            U = np.min([self.buffer[i]["size"],N,self.N_max])
            indices = np.random.permutation(self.buffer[i]["size"])[:U]

            X.append(self.buffer[i]["states"][indices])
            Y.append(self.buffer[i]["responses"][indices])

        return X,Y






        
'''
b = Ecperience_Buffer(4,0.1,4)

states = np.ones(4)
actions = 3
rewards = 0.5

b.record(np.ones(4),3,1)
b.record(np.ones(4),3,1)
b.record(np.ones(4),0,1)
b.end_of_game()
b.record(np.ones(4),3,1)
b.end_of_game()
b.record(np.ones(4)*3,3,1)
b.end_of_game()
b.record(np.ones(4)*5,3,1)
b.end_of_game()

X,Y = b.get_data(3)

print(X,Y)
'''