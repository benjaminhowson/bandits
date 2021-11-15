import numpy as np

class Bernoulli: 
    def __init__(self, parameters, expectations): 
        '''
        Description: 
            initialises bandit environment that returns binary rewards
            equal to zero or one from the bernoulli distribution

        Input: 
            parameters (dict): each key is an integer corresponding to an action
                               and values must contain parameters used by sample
                               function to generate the reward for the action
                               
            expectations (dict): each key is an integer correspoding to an action
                                 and values are the corresponding expected values
        '''
        self.parameters = parameters
        self.expectations = expectations

    def sample(self, action): 
        p = self.parameters[action]
        return np.random.binomial(n = 1, p = p, size = 1)[0]
