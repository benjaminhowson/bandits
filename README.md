# Bandits
The Bandits package is an easy-to-use framework for testing and comparing various bandit algorithms. Currently, the package implements numerous provably efficient algorithms for the problem, and more are to come! The Bernoulli environment included is a simple benchmark that allows one to experiment with the algorithms and investigate their performance. However, defining a custom environment is easy; see the environment class template in the final section!

# Problem Setting
Consider entering a casino. In front of you are five slot machines. You want to leave the casino with as much money as possible. It follows that you would like to find which of the five slot machines, on average, gives the highest reward.

<p align="center">
  <img src="https://cdn.analyticsvidhya.com/wp-content/uploads/2018/09/im_210.png" />
</p>

However, you have no information about any of the slot machines. How do you find the one with the highest expected reward while losing the least amount of money in the process? This problem is what bandit algorithms attempt to solve. 

# Using Bandits

## Custom Environments
To define the bandit, you need a function that samples rewards from the environment. Lets code up a simple environment with two actions. Suppose there are two slot machines at the casino; the first has an expected rewards equal to: 0.5; and the seconand has an expected reward equal to: 0.8

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
Once there is a function for sampling rewards, it is very easy to setup and run the bandit algorithm. All you need is the following lines of code:
```python
# setup the standard upper confidence bound algorithm to select one-hundred actions
agent = UCB1(sample = sample(), nactions = 2, nsamples = 100)

# let the agent interact with the environment by selecting one-hundred actions
agent.run()
```




# Custom Experiments
If you would like to run your own simulations and use the experiments function, you must create a class that stores the expected reward of each action so that one can plot the regret:

```python
class Custom:
  def __init__(self, parameters, expectations): 
    '''
    Input: 
      parameters (dict): each key is an integer corresponding to an action
                         and values must contain parameters used by sample
                         function to generate the reward for the action

      expectations (dict): each key is an integer correspoding to an action
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


