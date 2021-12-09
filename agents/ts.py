import numpy as np
from agents.agent import *

class BTS(Agent): 
    def __init__(self, sample, nactions): 
        '''
        Description
        -----------
        thompson sampling for bernoulli bandit environments

        Parameters
        ----------
        sample (func) - samples rewards
                      - input: time (int), action (int)
                      - output: {'time': [], 'action': [], 'reward': []}

        nactions (int) - number of actions available to the learner 
        '''
        super().__init__(sample, nactions) # initialise parent class

    def policy(self, time): 
        '''
        Description
        -----------
        selects the next action

        Parameters
        ----------
        time (int) - total number of steps taken
        '''
        best = np.inf
        highest = -np.inf
        for action in self.actions: 
            kwargs = self.parameters[action]
            betavalue = np.random.beta(**kwargs)
            if highest < betavalue: 
                best = action
                highest = betavalue
        return best

    def initialise(self): 
        '''
        Description
        -----------
        initialises the thompson sampling algorithm
        '''

        # initialise list of integers that corresponds to actions
        self.actions = [a for a in range(1, self.nactions + 1)]

        # initialise dictionary for storing actions and rewards
        self.history = {'time': [], 'action': [], 'reward': []}

        # initialise dictionary for storing parameter estimates
        self.parameters = {a: {} for a in self.actions}

        for action in self.actions: 
            self.parameters[action]['a'] = 1
            self.parameters[action]['b'] = 1

    def update_parameters(self, feedback): 
        '''
        Description
        -----------
        updates the parameter dictionary

        Parameters
        ----------
        feedback (dict) - {'time': [], 'action': [], 'reward': []}
        '''
        for i, action in enumerate(feedback['action']): 
            r = feedback['reward'][i]
            self.parameters[action]['a'] += r
            self.parameters[action]['b'] += 1 - r