import numpy as np
import agents.agent as agent

class UCB(agent.Agent): 
    def __init__(self, sample, nactions): 
        '''
        Description
        -----------
        parent class for the upper confidence bound multi-armed bandit 
        algorithms

        Parameters
        ----------
        sample (func) - samples rewards
                      - input: time (int), action (int)
                      - output: {'time': [], 'action': [], 'reward': []}

        nactions (int) - number of actions available to the learner 
        '''
        super().__init__(sample, nactions) # initialise parent class

    def bonus(self, time, action): 
        '''
        Description
        -----------
        placeholder function for the upper confidence bound bonus
        
        Parameters
        ----------
        time (int) - total number of steps taken
        action (int) - potential action at given time-step
        '''
        pass

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
            mu = self.parameters[action]['mu']
            ucb = mu + self.bonus(time, action)
            if highest < ucb: 
                best = action
                highest = ucb
        return best

    def initialise(self): 
        '''
        Description
        -----------
        initialises the upper confidence bound algorithm
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

class UCB1(UCB): 
    def __init__(self, sample, nactions):
        '''
        Description
        -----------
        class of functions for the classic upper confidence bound 
        bandit algorithm

        Parameters
        ---------- 
        sample (func) - samples rewards
                      - input: time (int), action (int)
                      - output: {'time': [], 'action': [], 'reward': []}

        nactions (int) - number of actions available to the learner 
        
        Reference
        --------- 
        Bandit Algorithms by Tor Lattimore & Csaba Szepsvari
        See Algorithm 3 for details.
        '''
        super().__init__(sample, nactions)
    
    def bonus(self, time, action): 
        delta = 1/pow(self.nsamples, 2)
        numerator = 2*np.log(1/delta)
        denominator = self.parameters[action]['n']
        return np.sqrt(numerator/denominator)

class AOUCB(UCB): 
    def __init__(self, sample, nactions):
        '''
        Description
        -----------
        class of functions for the asymptotically optimal upper 
        confidence bound bandit algorithm
        
        Parameters
        ----------
        sample (func) - samples rewards
                      - input: time (int), action (int)
                      - output: {'time': [], 'action': [], 'reward': []}

        nactions (int) - number of actions available to the learner 
        
        Reference
        --------- 
        Bandit Algorithms by Tor Lattimore & Csaba Szepsvari
        See Algorithm 6 for details.
        '''
        super().__init__(sample, nactions)
        
    def f(self, time): 
        return 1 + time*pow(np.log(time), 2)

    def bonus(self, time, action): 
        numerator = 2*np.log(self.f(time))
        denominator = self.parameters[action]['n']
        return np.sqrt(numerator/denominator)

class MOSS(UCB): 
    def __init__(self, sample, nactions):
        '''
        Description
        -----------
        minimax optimal strategy in the stochastic case upper 
        confidence bound bandit algorithm
        
        Parameters
        ----------
        sample (func) - samples rewards
                      - input: time (int), action (int)
                      - output: {'time': [], 'action': [], 'reward': []}

        nactions (int) - number of actions available to the learner 
        
        Reference
        ---------
            Bandit Algorithms by Tor Lattimore & Csaba Szepsvari
            See Algorithm 7 for details.
        '''
        super().__init__(sample, nactions)

    def plus(self, x): 
        return max(1, x)

    def bonus(self, time, action): 
        n = self.parameters[action]['n']
        x = self.nsamples/(self.nactions*n)
        return np.sqrt(4*np.log(self.plus(x))/n)

class ADA(UCB): 
    def __init__(self, sample, nactions):
        '''
        Description
        -----------
        adaptive upper confidence bound bandit algorithm
        
        Parameters
        ---------- 
        sample (func) - samples rewards
                      - input: time (int), action (int)
                      - output: {'time': [], 'action': [], 'reward': []}

        nactions (int) - number of actions available to the learner 
        
        Reference
        ---------
        Refining the Confidence Level for Optimistic Bandit 
        Strategies (2018)
        '''
        super().__init__(sample, nactions)

    def plus(self, x): 
        return max(1, x)

    def bonus(self, time, action): 
        d = 0.0
        n = self.parameters[action]['n']
        nt = {a: self.parameters[a]['n'] for a in self.actions}

        for a in nt.keys(): d += min(n, np.sqrt(n*nt[a]))
        return np.sqrt(2*np.log(self.plus(self.nsamples/d))/n)