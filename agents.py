import math
import random

class Agent: 
    def __init__(self, sample, nactions, nsamples): 
        '''
        Description: 
            parent class of all multi-armed bandit algorithms - contains
            shared methods and attributes

        Input: 
            sample (func): function that sample reward given an action
            nactions (int): number of available actions in environment 
            nsamples (int): number of rewards to sample from environment
        '''
        self.sample = sample
        self.nactions = nactions
        self.nsamples = nsamples

        # create list containing labels of all available actions
        self.actions = [a for a in range(1, nactions + 1)]

        # create dictionary for tracking sequences of actions and rewards
        self.history = {'time': [], 'action': [], 'reward': []}

        # create dictionary for storing parameters associated with actions
        self.parameters = {a: {} for a in self.actions}

    def reset(self): 
        self.history = {'time': [], 'action': [], 'reward': []}
        self.parameters = {a: {} for a in self.actions}
        self.initialise()
        

    def update_history(self, time, action, reward): 
        ''' 
        Description: 
            function that appends information at the given time-step 
            to the history dictionary
            
        Input: 
            time (int): current number of steps taken
            action (int): action taken at the given time index
            reward (int): reward received at the given time index
        '''
        self.history['time'].append(time)
        self.history['action'].append(action)
        self.history['reward'].append(reward)

class UCB(Agent):
    def __init__(self, sample, nactions, nsamples): 
        '''
        Description: 
            parent class of all multi-armed bandit algorithms using an 
            upper confidence bound strategy, containing shared methods 
            and attributes

        Input: 
            sample (func): function that sample reward given an action
            nactions (int): number of available actions in environment 
            nsamples (int): number of rewards to sample from environment
        '''
        super().__init__(sample, nactions, nsamples) # initialise agent
        self.initialise() # initialise the algorithm parameter dictionary 

    def run(self): 
        for time in range(1, self.nsamples + 1):
            action = self.policy(time)
            reward = self.sample(action)
            self.update(time, action, reward)

    def policy(self, time): 
        ''' 
        Description: 
            function that selects the action with the highest upper
            confidence bound
        '''
        action = math.inf
        highest = -math.inf
        for a in self.actions: 
            mu = self.parameters[a]['mu']
            uppercb = mu + self.bonus(action = a, time = time)
            if highest < uppercb: action, highest = a, uppercb
        return action

    def update(self, time, action, reward):
        ''' 
        Description: 
            function that updates the parameter and history dictionaries

        Input: 
            time (int): current number of steps taken
            action (int): action taken at the given time index
            reward (int): reward received at the given time index
        '''
        self.update_history(time, action, reward) 

        n = self.parameters[action]['n'] 
        mu = self.parameters[action]['mu']

        self.parameters[action]['n'] = n + 1
        self.parameters[action]['mu'] = mu + (reward - mu)/(n + 1)

    def initialise(self): 
        '''
        Description: 
            function that initialises the parameter dictionary with one 
            reward sample from each available action
        '''
        for a in self.actions: 
            rwd = self.sample(a)
            self.parameters[a]['n'] = 1
            self.parameters[a]['mu'] = rwd

class EGreedy(Agent): 
    def __init__(self, sample, nactions, nsamples, epsilon): 
        '''
        Description: 
            class of methods and attributes for the episilon-greedy
            approach to solving the multi-armed bandit problem.

        Input: 
            sample (func): function that sample reward given an action
            nactions (int): number of available actions in environment 
            nsamples (int): number of rewards to sample from environment
        '''
        super().__init__(sample, nactions, nsamples)
        self.initialise()
        self.epsilon = epsilon
    

    def run(self): 
        for time in range(1, self.nsamples + 1):
            action = self.policy()
            reward = self.sample(action)
            self.update(time, action, reward)
    
    def policy(self): 
        ''' 
        Description: 
            function that selects an action uniformly at random with
            probability ε and otherwise greedily selects the action 
            with the highest estimated expected reward
        '''
        p = random.uniform(0, 1)
        if random.uniform(0, 1) < self.epsilon: 
            action = random.randint(1, self.nactions)
        else: 
            action = math.inf
            highest = -math.inf
            for a in self.actions: 
                mu = self.parameters[a]['mu']
                if highest < mu: action, highest = a, mu

        return action

    def update(self, time, action, reward):
        ''' 
        Description: 
            function that updates the parameter and history dictionaries

        Input: 
            time (int): current number of steps taken
            action (int): action taken at the given time index
            reward (int): reward received at the given time index
        '''
        self.update_history(time, action, reward) 

        n = self.parameters[action]['n'] 
        mu = self.parameters[action]['mu']

        self.parameters[action]['n'] = n + 1
        self.parameters[action]['mu'] = mu + (reward - mu)/(n + 1)

    def initialise(self): 
        '''
        Description: 
            function that initialises the parameter dictionary with one 
            reward sample from each available action
        '''
        for a in self.actions: 
            rwd = self.sample(a)
            self.parameters[a]['n'] = 1
            self.parameters[a]['mu'] = rwd

class UCB1(UCB): 
    def __init__(self, sample, nactions, nsamples):
        '''
        Description:
            class of functions for the classic upper confidence bound 
            bandit algorithm

        Input: 
            sample (func): function that sample reward given an action
            nactions (int): number of available actions in environment 
            nsamples (int): number of rewards to sample from environment

        Reference: 
            Bandit Algorithms by Tor Lattimore & Csaba Szepsvari
            See Algorithm 3 for details.
        '''
        super().__init__(sample, nactions, nsamples)
    
    def bonus(self, action, time): 
        delta = 1/pow(self.nsamples, 2)
        numerator = 2*math.log(1/delta)
        denominator = self.parameters[action]['n']
        return math.sqrt(numerator/denominator)

class AOUCB(UCB): 
    def __init__(self, sample, nactions, nsamples):
        '''
        Description:
            class of functions for the asymptotically optimal upper confidence 
            bound bandit algorithm

        Input: 
            sample (func): function that sample reward given an action
            nactions (int): number of available actions in environment 
            nsamples (int): number of rewards to sample from environment

        Reference: 
            Bandit Algorithms by Tor Lattimore & Csaba Szepsvari
            See Algorithm 6 for details.
        '''
        super().__init__(sample, nactions, nsamples)
        
    def f(self, time): 
        return 1 + time*pow(math.log(time), 2)

    def bonus(self, action, time): 
        numerator = 2*math.log(self.f(time))
        denominator = self.parameters[action]['n']
        return math.sqrt(numerator/denominator)

class MOSS(UCB): 
    def __init__(self, sample, nactions, nsamples):
        '''
        Description:
            class of functions for the minimax optimal strategy in the stochastic case
            upper confidence bound bandit algorithm

        Input: 
            sample (func): function that sample reward given an action
            nactions (int): number of available actions in environment 
            nsamples (int): number of rewards to sample from environment

        Reference: 
            Bandit Algorithms by Tor Lattimore & Csaba Szepsvari
            See Algorithm 7 for details.
        '''
        super().__init__(sample, nactions, nsamples)

    def plus(self, x): 
        return max(1, x)

    def bonus(self, action, time): 
        n = self.parameters[action]['n']
        x = self.nsamples/(self.nactions*n)
        return math.sqrt(4*math.log(self.plus(x))/n)

class ADA(UCB): 
    def __init__(self, sample, nactions, nsamples):
        '''
        Description:
            class of functions for the minimax optimal strategy in the stochastic case
            upper confidence bound bandit algorithm

        Input: 
            sample (func): function that sample reward given an action
            nactions (int): number of available actions in environment 
            nsamples (int): number of rewards to sample from environment

        Reference: 
            Refining the Confidence Level for Optimistic Bandit Strategies (2018)
        '''
        super().__init__(sample, nactions, nsamples)

    def plus(self, x): 
        return max(1, x)

    def bonus(self, action, time): 
        d = 0.0
        n = self.parameters[action]['n']
        nt = {a: self.parameters[a]['n'] for a in self.actions}

        for a in nt.keys(): d += min(n, math.sqrt(n*nt[a]))
        return math.sqrt(2*math.log(self.plus(self.nsamples/d))/n)

class TS(Agent): 
    def __init__(self, sample, nactions, nsamples): 
        '''
        Description: 
            class of functions for the thompson sampling bandit algorithm

        Input: 
            sample (func): function that sample reward given an action
            nactions (int): number of available actions in environment 
            nsamples (int): number of rewards to sample from environment

        Reference: 
            A Tutorial on Thompson Sampling by Daniel J. Russo et al. 
            See Algorithm 4 for details.
        '''
        super().__init__(sample, nactions, nsamples)
        self.initialise()

    def run(self): 
        for time in range(1, self.nsamples + 1):
            action = self.policy()
            reward = self.sample(action)
            self.update(time, action, reward)
    
    def policy(self): 
        ''' 
        Description: 
            function that selects an action by sampling reward from the
            beta distribution
        '''
        action = math.inf
        highest = -math.inf

        for a in self.actions: 
            value = random.betavariate(**self.parameters[a])
            if highest < value: action, highest = a, value

        return action

    def update(self, time, action, reward):
        ''' 
        Description: 
            function that updates the parameter and history dictionaries

        Input: 
            time (int): current number of steps taken
            action (int): action taken at the given time index
            reward (int): binary reward received at the given time index
        '''
        self.update_history(time, action, reward) 
        self.parameters[action]['alpha'] += reward
        self.parameters[action]['beta'] += 1 - reward

    def initialise(self): 
        '''
        Description: 
            function that initialises the parameter dictionary
        '''
        for a in self.actions: 
            self.parameters[a] = {'alpha': 1, 'beta': 1}