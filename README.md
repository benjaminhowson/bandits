# Bandits

The Bandits package is an easy-to-use framework for testing and comparing various bandit algorithms. The BernoulliBandit environment is a simple benchmark that allows one to experiment with the algorithms and investigate their performance. However, defining a custom environment is easy; all the custom environment-class needs is a method: 'sample(action)', which returns a numeric reward associated with the given action. 
