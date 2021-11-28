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
        for time in range(1, nsamples + 1):
            action = self.policy(time) # select an action
            feedback = self.sample(time, action) # observe feedback

            self.update_history(time, action) # update history
            self.update_parameters(feedback) # update parameters

        if reset:
            history = self.history 
            self.initialise()
            return history


    def sort_history(self): 
        '''
        Description
        -----------
        sorts all lists in the history dictionary according to time
        '''
        pass

    def update_history(self, time, action): 
        '''
        Description
        -----------
        updates the history dictionary using feedback from environment

        Parameters
        ----------
        feedback (dict) - {'time': [], 'action': [], 'reward': []}
        '''
        self.history['time'].append(time)
        self.history['action'].append(action)