# Bandits
The Bandits package is an easy-to-use framework for testing and comparing various bandit algorithms. Currently, the package implements numerous provably efficient algorithms for the problem, and more are to come! The Bernoulli environment included is a simple benchmark that allows one to experiment with the algorithms and investigate their performance. However, defining a custom environment is easy; see the environment class template in the final section!

## Algorithms
Currently, the package implements the following algorithms
* UCB1
* MOSS
* AOUCB
* ADA-UCB
* Epsilon-Greedy
* Thompson Sampling (Bernoulli rewards only)

# Problem Setting
Consider entering a casino. In front of you are five slot machines. You want to leave the casino with as much money as possible. It follows that you would like to find which of the five slot machines, on average, gives the highest reward.

<p align="center">
  <img src="https://cdn.analyticsvidhya.com/wp-content/uploads/2018/09/im_210.png" />
</p>

However, you have no information about any of the slot machines. How do you find the one with the highest expected reward while losing the least amount of money in the process? This problem is what bandit algorithms attempt to solve. 

# Using Bandits
## Sample Function
To define the bandit, all you need is a function that samples a reward for each action. 

Below is an example of a simple environment with two actions whose expected rewards are equal to 0.5 and 0.8:

```python
def sample(action): 
  '''
  Input: 
    action (int): selection of the agent
  '''
  if action == 1: 
    # code to get a reward for the first action
    return np.random.binomial(1, 0.5, 1)[0]
  elif action == 2: 
    # code to get a reward for the second action
    return np.random.binomial(1, 0.8, 1)[0]
  else: 
    # code to raise errors upon selection an invalid action
    raise ValueError('Error: only two available actions')
```

## Agents
Once there is a function for sampling rewards, it is easy to set up and run the bandit algorithm. All you need is the following lines of code:
```python
from agents import *

# setup the standard upper confidence bound algorithm to select one-hundred actions
agent = UCB1(sample = sample, nactions = 2, nsamples = 100)

# let the agent interact with the environment by selecting one-hundred actions
history = agent.run()
```
The variable ```history``` is a dictionary containing the ordered sequence of actions and the corresponding rewards. Also, ```history``` is a class variable, meaning you can access it via: ```agent.history```




# Experiments

## Bernoulli Bandit Experiment
Included is a standard benchmark environment for testing multi-armed bandit algorithms. Consider the example above; an environment with two actions whose reward distributions are: Bernoulli(0.5) and Bernoulli(0.8), respectively. 

We can compare various algorithms using the ```experiment()``` function.
```python
from experiments import *
from environments import *

parameters = {1: 0.5, 2: 0.8} # distributional parameters for each action
expectations = {1: 0.5, 2: 0.8} # expected reward for each action

environment = Bernoulli(parameters, expectations) # create bernoulli bandit environment

# compare thompson sampling to various upper confidence bound strategies
learners = {'TS': TS(sample = environment.sample, nactions = 2, nsamples = 1000), 
            'UCB': UCB1(sample = environment.sample, nactions = 2, nsamples = 1000), 
            'MOSS': MOSS(sample = environment.sample, nactions = 2, nsamples = 1000), 
            'AOUCB': AOUCB(sample = environment.sample, nactions = 2, nsamples = 1000), 
            'ADA-UCB': ADA(sample = environment.sample, nactions = 2, nsamples = 1000)}

# perform the experiment using monte-carlo simulation using one-hundred iterations
output = experiment(iterations = 100, environment = environment, learners = learners)

# visualise the results
plot(data = output, width = 0.5)

```

<p align="center">
  <img src="images/experiment.png" />
</p>


## Custom Experiments
If you would like to run simulations using different distributions and make use of: ```experiment()```  and ```plot()```, you must create a class using the following template:

```python
class Custom:
  def __init__(self, parameters, expectations): 
    '''
    Input: 
      parameters (dict): each key is an integer corresponding to an action
                         and values must contain parameters used by the
                         function to generate the reward for the action

      expectations (dict): each key is an integer corresponding to an action
                           and values are the corresponding expected values
    '''
    self.parameters = parameters
    self.expectations = expectations
    
  def samples(self, action): 
    '''
    Input: 
      action (int): selection of the agent
    '''
    return function(parameters[action])
```
We refer the user to the ```environments.py``` file of this repository for the Bernoulli bandit environment implementation using the above template.


