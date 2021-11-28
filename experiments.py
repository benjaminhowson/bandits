import numpy as np
import matplotlib.pyplot as plt

def experiment(learners, nsamples, niterations, environment): 
    '''
    Description
    -----------
    runs simulation experiment for empirically evaluating the
    regret of the chosen bandit algorithms

    Parameters
    ----------
    nsamples (int) - total number of rewards to sample
    niterations (int) - total number of experiments to run
    environment (obj) - parameterised environment for algorithm
    '''
    mu = environment.expectations
    k = len(mu.keys()); opt = max(mu, key = mu.get)

    zeros = np.zeros((nsamples, niterations))
    performance = {l: zeros.copy() for l in learners.keys()}

    for iter in range(niterations): 
        for learner in learners.keys(): 
            agent = learners[learner]
            history = agent.run(nsamples, reset = True)

            for t in range(nsamples): 
                if t == 0: previous = 0.00
                else: previous = performance[learner][t - 1, iter]

                action = history['action'][t]
                regret = mu[opt] - mu[action]
                performance[learner][t, iter] = previous + regret
            
    return {learner: {'mean': np.mean(performance[learner], axis = 1), 
                      'std': np.std(performance[learner], axis = 1)}
                       for learner in performance.keys()}
    

def plot(data, width = 1.0): 
    '''
    Parameters
    ----------
    data (dict) - output from the experiments function
    zvalue (int) - defines the width of the confidence intervals
    '''
    fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize = (12, 9))
    cols = ['tab:blue', 'tab:orange', 'tab:green', 
            'tab:red', 'tab:purple', 'tab:brown', 
            'tab:pink', 'tab:grey', 'tab:olive', 
            'tab:cyan']

    for col, learner in enumerate(data.keys()): 
        std = np.array(data[learner]['std'])
        mean = np.array(data[learner]['mean'])
        upper = mean + width*std; lower = mean - width*std

        time = np.array([t + 1 for t in range(std.shape[0])])
        ax.plot(time, mean, c = cols[col], label = learner)
        ax.fill_between(time, upper, lower, 
                        color = cols[col], 
                        alpha = 0.15) 
        
    ax.set_ylabel('Cumulative Regret', fontsize = 12)
    ax.set_xlabel('Time', fontsize = 12)

    plt.legend(frameon = False, loc = 'upper left')
    plt.show()
