import numpy as np
import agents.agent as agent

class EG(agent.Agent): 
    def __init__(self, sample, nactions, epsilon = 0.05): 
        '''
        Description
        -----------
        epsilon-greedy multi-armed bandit algorithm

        Parameters
        ----------
        sample (func) - samples rewards
                      - input: time (int), action (int)
                      - output: {'time': [], 'action': [], 'reward': []}

        nactions (int) - number of actions available to the learner 
        epsilon (float) - probability of select an action at random
        '''
        self.epsilon = epsilon # initialise algorithm parameter
        super().__init__(sample, nactions) # initialise parent class

    def policy(self, time): 
        '''
        Description
        -----------
        selects the next action

        Parameters
        ----------
        time (int) - total number of steps taken
        '''
        uniform = np.random.uniform(low = 0, high = 1)
        if uniform < self.epsilon: 
            return np.random.choice(self.actions)

        else: 
            best = np.inf
            highest = -np.inf
            for action in self.actions: 
                mu = self.parameters[action]['mu']
                if highest < mu: 
                    best = action
                    highest = mu
            return best

    def initialise(self): 
        '''
        Description
        -----------
        initialises the epsilon greedy algorithm
        '''
        # initialise list of integers that corresponds to actions
        self.actions = [a for a in range(1, self.nactions + 1)]

        # initialise dictionary for storing actions and rewards
        self.history = {'time': [], 'action': [], 'reward': []}

        # initialise dictionary for storing parameter estimates
        self.parameters = {a: {} for a in self.actions}

        for action in self.actions: 
            self.parameters[action]['n'] = 1
            self.parameters[action]['mu'] = 1.00

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
            n = self.parameters[action]['n'] 
            mu = self.parameters[action]['mu']

            self.parameters[action]['n'] = n + 1
            self.parameters[action]['mu'] = mu + (r - mu)/(n + 1)