# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


import mdp, util

from learningAgents import ValueEstimationAgent

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0
        # Write value iteration code here
        "*** YOUR CODE HERE ***"
        # create a dictionary of (state, best possible actions) and store it internally
        # to allow O(1) retrieval of an action given its state
        self.bestActions = {}
 
        # iterate the given number of times
        for i in range(0, self.iterations):
          # create a temporary dictionary to save the newly computed values
          tempDict = util.Counter()
          
          # iterate over all the avaliable states withing the game board
          for state in self.mdp.getStates():
            # get the best possible action for this state
            bestAction = self.computeActionFromValues(state)
            # get the Q-Value of this state, only if the bestAction isn't None
            qValue = 0.0
            if bestAction != None:
              qValue = self.computeQValueFromValues(state, bestAction)
            # add the new value to the temporary dictionary 
            tempDict[state] = qValue
            # add the new best action to the list of best actions
            self.bestActions[state] = bestAction

          # update the value of state in self.value
          self.values = tempDict



    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        "*** YOUR CODE HERE ***"
        # get the list of transitons states and probabilities 
        tpList = self.mdp.getTransitionStatesAndProbs(state, action)
        # initialize the Q-Value to zero
        qValue = 0.0

        # iterate over all next states
        for t in tpList:
          # extract the next state and its probability 
          (nextState, prob) = t
          # get the reward for moving into nextState from state via action
          reward = self.mdp.getReward(state, action, nextState)
          # get the value of next state
          nsValue = self.values[nextState]

          # add the value from Bellman Backup equation to qValue
          qValue += prob*(reward + self.discount*nsValue)

        # return the computed qValue
        return qValue


    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        # get all the possible actions from this state
        actions = self.mdp.getPossibleActions(state)

        # if state was the terminal state return None
        if self.mdp.isTerminal(state) or len(actions) == 0:
          return None
    
        # initialize the maximum value to be -infinity
        maxValue = -1*float("inf")
        # initialize the maximum action to be None
        maxAction = None

        # loop over all actions
        for a in actions:
          # get the qValue of this action
          actionValue = self.computeQValueFromValues(state, a)

          # update if the current qValue was better than max
          if actionValue >= maxValue:
            maxValue = actionValue
            maxAction = a

        # return the best action
        return maxAction

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.bestActions[state]

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)

