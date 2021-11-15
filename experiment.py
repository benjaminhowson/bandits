from agents import *
from environments import *

import matplotlib.pyplot as plt


def experiment(iterations, environment, learners): 
    mu = environment.expectations
    k = len(mu.keys()); opt = max(mu, key = mu.get)

    performance = {l: np.zeros(shape = (learners[l].nsamples, iterations)) for l in learners.keys()}

    for iter in range(iterations): 
        for learner in learners.keys(): 
            agent = learners[learner]
            history = agent.run()

            for t in range(agent.nsamples): 
                if t == 0: previous = 0.00
                else: previous = performance[learner][t - 1, iter]

                action = history['action'][t]
                regret = mu[opt] - mu[action]
                performance[learner][t, iter] = previous + regret
            
            learners[learner].reset()

    output = {learner: 
                {'mean': np.mean(performance[learner], axis = 1), 
                'std': np.std(performance[learner], axis = 1)}
                for learner in performance.keys()}
        
    return output

def plot(data): 
    fig, ax = plt.subplots(nrows = 1, ncols = 1)
    cols = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 
            'tab:brown', 'tab:pink', 'tab:grey', 'tab:olive', 'tab:cyan']

    for col, learner in enumerate(data.keys()): 
        std = np.array(data[learner]['std'])
        mean = np.array(data[learner]['mean'])
        upper = mean + std; lower = mean - std

        time = np.array([t + 1 for t in range(std.shape[0])])
        ax.plot(time, mean, c = cols[col], label = learner)
        ax.fill_between(time, upper, lower, color = cols[col], alpha = 0.15) 


    ax.set_ylabel('Cumulative Regret')
    ax.set_xlabel('Time')

    plt.legend(frameon = False, loc = 'upper left')
    plt.show()





    