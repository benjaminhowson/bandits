import numpy as np

class Bernoulli: 
    def __init__(self, probabilities): 
        '''
        Description: 
            initialises bandit environment that returns binary rewards
            equal to zero or one from the bernoulli distribution

        Input: 
            probabilities (dict): each key is an integer representing an action and 
                                  corresponding value is the probability of success
        '''
        self.probabilities = probabilities

    def sample(self, action): 
        p = self.probabilities[action]
        return np.random.binomial(n = 1, p = p, size = 1)[0]
