import numpy as np

class Bernoulli: 
    def __init__(self, parameters, expectations): 
        '''
        Description
        -----------
        initialises bandit environment that returns binary rewards
        equal to zero or one from the bernoulli distribution
        
        Parameters 
        ---------- 
        parameters (dict) - {1: kwargs, ..., K: kwargs}              
        expectations (dict) - {1: mean, ..., K: mean}
        '''
        self.parameters = parameters
        self.expectations = expectations

    def sample(self, time, action): 
        p = self.parameters[action]
        r = np.random.binomial(n = 1, p = p)
        return {'time': [time], 'action': [action], 'reward': [r]}

