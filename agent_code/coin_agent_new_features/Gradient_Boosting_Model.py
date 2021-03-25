from xgboost import XGBRegressor
import numpy as np
import pickle

'''
Tutorial:
https://machinelearningmastery.com/extreme-gradient-boosting-ensemble-in-python/#:~:text=As%20such,%20XGBoost%20is%20an%20algorithm,%20an%20open-source,titled%20%E2%80%9C%20XGBoost:%20A%20Scalable%20Tree%20Boosting%20System.%E2%80%9D
'''

class GBM_Model():
    def __init__(self,params = {}):

        self.ensembles = []

        #Add a single Ensemble for each of the possible actions
        for i in range(6):
            self.ensembles.append(XGBRegressor(**params))

    def __call__(self,X,bombing = True):
        '''
        Parameters:
            X:          Feature vector of size (D,)
            bombing:    Is the agent allowed to use bombs?          default: True

        Returns:
            p:          Predictions vector of size (6,)
        '''
        
        #Is the agent allowed to use bombs?
        if bombing == True: u = 6
        else: u = 5

        #Get the predicted Q value for each of the possible actions
        p = np.zeros(6)

        for i in range(u):
            p[i] = self.ensembles[i].predict(X.reshape(1,-1))
        
        return p
        
    def train(self,X,Y):
        '''
        X and contain a single array for each of the six actions.
        '''

        for i in range(6):
            #Only train if there are training instances:
            if len(X[i]) == 0: continue
            self.ensembles[i].fit(X[i],Y[i])