from os import environ
from agents import *
from environments import *

import matplotlib.pyplot as plt


def experiment(iterations, environment, learners): 
    mu = environment.expectations
    k = len(mu.keys()); opt = max(mu, key = mu.get)

    performance = {l: np.zeros(shape = (learners[l].nsamples, iterations)) for l in learners.keys()}

    for iter in range(iterations): 
        for learner in learners.keys(): 
            agent = learners[learner]; agent.run()

            for t in range(nsamples): 
                if t == 0: previous = 0.00
                else: previous = performance[learner][t - 1, iter]

                action = agent.history['action'][t]
                regret = mu[opt] - mu[action]
                performance[learner][t, iter] = previous + regret
            
            learners[learner].reset()

    output = {learner: 
                {'mean': np.mean(performance[learner], axis = 1), 
                'std': np.std(performance[learner], axis = 1)}
                for learner in performance.keys()}
        
    return output

nsamples = 1000
iterations = 100
environment = Bernoulli({1: 0.5, 2: 0.8, 3: 0.2})
learners = {'TS': TS(sample = environment.sample, nactions = 3, nsamples = nsamples), 
            'UCB': UCB1(sample = environment.sample, nactions = 3, nsamples = nsamples)}

output = experiment(iterations, environment, learners)


fig, ax = plt.subplots(nrows = 1, ncols = 1)

time = np.array([t + 1 for t in range(nsamples)])
cols = {'TS': 'tab:blue', 'UCB': 'tab:orange'}

for learner in output.keys(): 
    std = np.array(output[learner]['std'])
    mean = np.array(output[learner]['mean'])
    upper = mean + 1.96*std; lower = mean - 1.96*std

    ax.plot(time, mean, c = cols[learner], label = learner)
    ax.fill_between(time, upper, lower, color = cols[learner], alpha = 0.15) 


ax.set_ylabel('Cumulative Regret')
ax.set_xlabel('Time')

plt.legend(frameon = False, loc = 'upper left')
plt.show()




    