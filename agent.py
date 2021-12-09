class Agent: 
    def __init__(self, sample, nactions): 
        '''
        Description
        -----------
        parent class for all multi-armed bandit algorithms

        Parameters
        ----------
        sample (func) - samples rewards
                      - input: time (int), action (int)
                      - output: {'time': [], 'action': [], 'reward': []}

        nactions (int) - number of actions available to the learner 
        '''
        self.sample = sample # initialise number of samples
        self.nactions = nactions # initialise number of actions
        self.initialise() # initialise the learner

    def policy(self, time): 
        '''
        Description
        -----------
        placeholder function for selecting the next action

        Parameters
        ----------
        time (int) - total number of steps taken
        '''
        pass
    
    def initialise(self): 
        '''
        Description
        -----------
        placeholder function for initialising the parameters algorithm 
        '''
        pass

    def update_parameters(self, feedback): 
        '''
        Description
        -----------
        placeholder function for updating the parameters algorithm 
        '''
        pass

    def run(self, nsamples, reset = False): 
        '''
        Description
        -----------
        runs the multi-armed bandit algorithm in the environment for the
        given number of steps

        Parameters
        ----------
        nsamples (int) - number of steps to take in the environment
        reset (bool) - indicate whether to reset upon finishing
        '''
        self.nsamples = nsamples
        self.history['reward'] = [float('NaN')]*nsamples

        for time in range(1, nsamples + 1):
            action = self.policy(time) # select an action
            feedback = self.sample(time, action) # observe feedback

            self.update_parameters(feedback) # update parameters
            self.update_history(time, action, feedback) # update history

        if reset:
            history = self.history 
            self.initialise()
            return history
        else: 
            return self.history

    def update_history(self, time, action, feedback): 
        '''
        Description
        -----------
        updates the history dictionary using feedback from environment

        Parameters
        ----------
        time (int) - time of action
        action (int) - selection of the agent
        feedback (dict) - {'time': [], 'action': [], 'reward': []}
        '''
        self.history['time'].append(time)
        self.history['action'].append(action)
        for idx, t in enumerate(feedback['time']): 
            reward = feedback['reward'][idx]
            self.history['reward'][t - 1] = reward
