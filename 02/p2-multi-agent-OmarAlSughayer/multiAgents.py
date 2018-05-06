# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for 
# educational purposes provided that (1) you do not distribute or publish 
# solutions, (2) you retain this notice, and (3) you provide clear 
# attribution to UC Berkeley, including a link to 
# http://inst.eecs.berkeley.edu/~cs188/pacman/pacman.html
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero 
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and 
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random, util
import math 

from game import Agent

class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        # scores = [better(gameState.generatePacmanSuccessor(action) for action in legalMoves]

        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        # choose the best move
        return legalMoves[scores.index(bestScore)]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()

        # lowest possible value to losing states
        if successorGameState.isLose():
          return 0

        # food grid and list
        newFood = successorGameState.getFood()
        foodList = newFood.asList()

        # ghostsState lists and ghost positions
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [gs.scaredTimer for gs in newGhostStates]
        newGhostPos = [gs.getPosition() for gs in newGhostStates]

        "*** YOUR CODE HERE ***"
        # find the total distance to food, then add one to avoid dividisoin by zero
        dFood = self.findTotalDistance(newPos, foodList) + 1

        # get the size of the board
        boardSize = newFood.width + newFood.height

        # find the minimum distance to a ghost
        mdGhost = self.findMinDistance(newPos, newGhostPos)
        
        # if the next position puts you right next to a ghost then
        # don't take it becaues the ghost will advance to you AND FREAKING MURDER YOU
        if mdGhost <= 1:
          mdGhost = 0

        # find the difference between the current and next score
        ds = successorGameState.getScore() - currentGameState.getScore()

        return mdGhost*(1.0/dFood)*abs(ds)

    def findTotalDistance(self, origin, targets):
      """
      finds the total distance from the origin to every other target
      """
      total = 0
      (oX, oY) = origin
      # adds the change in x and y to the total distance
      for t in targets:
        (tX, tY) = t
        distance = abs(tX - oX) + abs(tY - oY)
        total += distance

      # return the total distance
      return total 

    def findMinDistance(self, origin, targets):
      """
      finds the minimum distance between origin and all the targets
      """
      # return -1 if there are no targets 
      if len(targets) == 0:
        return -1

      # fence post problem
      (oX, oY) = origin
      (tX, tY) = targets[0]
      minimum = abs(tX - oX) + abs(tY - oY)
      
      # loops over all the targets
      for t in targets:
        (tX, tY) = t
        distance = abs(tX - oX) + abs(tY - oY)
        # potentially change min
        if distance < minimum:
          minimum = distance

      # return the total distance
      return minimum

def scoreEvaluationFunction(currentGameState):
    """
      This default evaluation function just returns the score of the state.
      The score is the same one displayed in the Pacman GUI.

      This evaluation function is meant for use with adversarial search agents
      (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
      This class provides some common elements to all of your
      multi-agent searchers.  Any methods defined here will be available
      to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

      You *do not* need to make any changes here, but you can if you want to
      add functionality to all your adversarial search agents.  Please do not
      remove anything, however.

      Note: this is an abstract class: one that should not be instantiated.  It's
      only partially specified, and designed to be extended.  Agent (game.py)
      is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action from the current gameState using self.depth
          and self.evaluationFunction.

          Here are some method calls that might be useful when implementing minimax.

          gameState.getLegalActions(agentIndex):
            Returns a list of legal actions for an agent
            agentIndex=0 means Pacman, ghosts are >= 1

          gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action

          gameState.getNumAgents():
            Returns the total number of agents in the game
        """
        "*** YOUR CODE HERE ***"

        # call a ____ function with v(gameState, agentIndex, depth)
        (chosenAction, valueOfAction) = self.v(gameState, 0, self.depth)

        # return the chosen action
        return chosenAction

    def v(self, gameState, agentIndex, cDepth):
      """
      performs a minmax tree search with the given gamestate as the root and the given
      agentIndex as the frist index to move followed by only the agents proceeding it within
      this level of the tree, until depth cDepth
      """

      # if we already iterated over all the agents, start over and decrease the depth
      if agentIndex >= gameState.getNumAgents():
        agentIndex = 0 # pacman index
        cDepth -= 1

      # base case if required depth have been reached
      if cDepth == 0:
        return self.evaluationFunction(gameState)

      # other than that call minOrMaxValue with max funciton for pacman agent 
      if agentIndex == 0:
        return self.minOrMaxValue(gameState, agentIndex, cDepth, max)
      else: # and call min-minOrMaxValue for with min function non-acman agents
        return self.minOrMaxValue(gameState, agentIndex, cDepth, min)

    def minOrMaxValue(self, gameState, agentIndex, cDepth, func):
      """
      performs a min or a max value search on all children of the given gameState
      depending on the given function
      """

      # if there are no legal actions left, make into a leaf
      if len(gameState.getLegalActions(agentIndex)) == 0:
        return self.evaluationFunction(gameState)

      # if the function passed was max then the initial value should be -infinity
      # if the function passed was min then the initial value should be +infinity
      initValue = -1*func(-1*float("inf"), float("inf"))

      # the initial value is a tuple of no action (none) and the initial value
      (bestAction, bestValue) = (None, initValue)

      # loop over all the children and find the child with the best value
      for childAction in gameState.getLegalActions(agentIndex):
        # get the state that proceeds from taking this action
        childState = gameState.generateSuccessor(agentIndex, childAction)

        # get the value of the child state
        childValue = self.v(childState, agentIndex + 1, cDepth)

        # the value of the child could either be a float or a tuple, ensure that it is a float
        if isinstance(childValue, tuple):
            (tempAction, tempValue) = childValue
            childValue = tempValue

        # find the new best value according to the given function
        funcValue = func(bestValue, childValue)

        # if the new foind value is better than the so-far-best value, store it
        if funcValue != bestValue:
            bestValue = childValue
            bestAction = childAction 
        
      # return the best answer found
      return (bestAction, bestValue)


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the alpha-beta action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        alpha = -1*float("inf")
        beta = float("inf")
        # call a ____ function with v(gameState, agentIndex, depth, alpha, beta)
        (chosenAction, valueOfAction) = self.v(gameState, 0, self.depth, alpha, beta)

        # return the chosen action
        return chosenAction

    def v(self, gameState, agentIndex, cDepth, alpha, beta):
      """
      performs an alpha-beta tree search with the given gamestate as the root and the given
      agentIndex as the frist index to move followed by only the agents proceeding it within
      this level of the tree, until depth cDepth
      """

      # if we already iterated over all the agents, start over and decrease the depth
      if agentIndex >= gameState.getNumAgents():
        agentIndex = 0 # pacman index
        cDepth -= 1

      # base case if required depth have been reached
      if cDepth == 0:
        return self.evaluationFunction(gameState)

      # other than that call alphaBetaValue with max funciton for pacman agent 
      if agentIndex == 0:
        return self.alphaBetaValue(gameState, agentIndex, cDepth, beta, alpha, max)
      else: # and call alphaBetaValue for with min function non-acman agents
        return self.alphaBetaValue(gameState, agentIndex, cDepth, alpha, beta, min)

    def alphaBetaValue(self, gameState, agentIndex, cDepth, limiterX, limiterY, func):
      """
      performs an alpha-beta value search on all children of the given gameState
      depending on the given function
      """

      # if there are no legal actions left, make into a leaf
      if len(gameState.getLegalActions(agentIndex)) == 0:
        return self.evaluationFunction(gameState)

      # if the function passed was max then the initial value should be -infinity
      # if the function passed was min then the initial value should be +infinity
      initValue = -1*func(-1*float("inf"), float("inf"))

      # the initial value is a tuple of no action (none) and the initial value
      (bestAction, bestValue) = (None, initValue)

      # loop over all the children and find the child with the best value
      for childAction in gameState.getLegalActions(agentIndex):
        # get the state that proceeds from taking this action
        childState = gameState.generateSuccessor(agentIndex, childAction)

        # since the current node have not been proned we can assume the following
        alpha = min(limiterX, limiterY)
        beta = max(limiterX, limiterY)

        # get the value of the child state
        childValue = self.v(childState, agentIndex + 1, cDepth, alpha, beta)

        # the value of the child could either be a float or a tuple, ensure that it is a float
        if isinstance(childValue, tuple):
            (tempAction, tempValue) = childValue
            childValue = tempValue

        # find the new best value according to the given function
        funcValue = func(bestValue, childValue)

        # if the new foind value is better than the so-far-best value, store it
        if funcValue != bestValue:
            bestValue = childValue
            bestAction = childAction 
        
        # attempt to prune
        pruneValue = func(bestValue, limiterX)
        if pruneValue == bestValue and pruneValue != limiterX:
          return (bestAction, bestValue)

        # update the value for beta
        limiterY = func(bestValue, limiterY)

      # return the best answer found
      return (bestAction, bestValue)

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        "*** YOUR CODE HERE ***"
        # call a ____ function with v(gameState, agentIndex, depth)
        (chosenAction, valueOfAction) = self.v(gameState, 0, self.depth)

        # return the chosen action
        return chosenAction

    def v(self, gameState, agentIndex, cDepth):
      """
      performs an expected value tree search with the given gamestate as the root and the given
      agentIndex as the frist index to move followed by only the agents proceeding it within
      this level of the tree, until depth cDepth
      """
      # if we already iterated over all the agents, start over and decrease the depth
      if agentIndex >= gameState.getNumAgents():
        agentIndex = 0 # pacman index
        cDepth -= 1

      # base case if required depth have been reached
      if cDepth == 0:
        return self.evaluationFunction(gameState)

      # other than that call alphaBetaValue with max funciton for pacman agent 
      if agentIndex == 0:
        return self.maxValue(gameState, agentIndex, cDepth)
      else: # and call alphaBetaValue for with min function non-acman agents
        return self.expectedValue(gameState, agentIndex, cDepth)

    def maxValue(self, gameState, agentIndex, cDepth):
      """
      performs a max value search on all children of the given gameState
      """

      # if there are no legal actions left, make into a leaf
      if len(gameState.getLegalActions(agentIndex)) == 0:
        return self.evaluationFunction(gameState)

      # if the function passed was max then the initial value should be -infinity
      # if the function passed was min then the initial value should be +infinity
      # initValue = -1*func(-1*float("inf"), float("inf"))

      # the initial value is a tuple of no action (none) and the initial value
      (bestAction, bestValue) = (None, -1*float("inf"))

      # loop over all the children and find the child with the best value
      for childAction in gameState.getLegalActions(agentIndex):
        # get the state that proceeds from taking this action
        childState = gameState.generateSuccessor(agentIndex, childAction)

        # get the value of the child state
        childValue = self.v(childState, agentIndex + 1, cDepth)

        # the value of the child could either be a float or a tuple, ensure that it is a float
        if isinstance(childValue, tuple):
            (tempAction, tempValue) = childValue
            childValue = tempValue

        # find the new best value according to the given function
        funcValue = max(bestValue, childValue)

        # if the new foind value is better than the so-far-best value, store it
        if funcValue != bestValue:
            bestValue = childValue
            bestAction = childAction 

      # return the best answer found
      return (bestAction, bestValue)

    def expectedValue(self, gameState, agentIndex, cDepth):
      """
      performs an expected value search on all children of the given gameState
      """

      # if there are no legal actions left, make into a leaf
      if len(gameState.getLegalActions(agentIndex)) == 0:
        return self.evaluationFunction(gameState)

      # if the function passed was max then the initial value should be -infinity
      # if the function passed was min then the initial value should be +infinity
      # initValue = -1*func(-1*float("inf"), float("inf"))

      # the initial value is a tuple of no action (none) and the initial value
      (bestAction, expectedValue) = (None, 0)

      # the probability for every action
      p = 1.0/len(gameState.getLegalActions(agentIndex))

      # loop over all the children and find the child with the best value
      for childAction in gameState.getLegalActions(agentIndex):
        # get the state that proceeds from taking this action
        childState = gameState.generateSuccessor(agentIndex, childAction)

        # get the value of the child state
        childValue = self.v(childState, agentIndex + 1, cDepth)

        # the value of the child could either be a float or a tuple, ensure that it is a float
        if isinstance(childValue, tuple):
            (tempAction, tempValue) = childValue
            childValue = tempValue

        # add the value of the child*its probability to the expected value
        expectedValue += p*childValue
        # the best action does not matter since it will be calculated at the root

      # return the expected value of the node
      return (bestAction, expectedValue)

def betterEvaluationFunction(currentGameState):
  """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
  """
  "*** YOUR CODE HERE ***"

  # lowest possible value to losing states
  if currentGameState.isLose():
    return 0

  # food grid and list
  newFood = currentGameState.getFood()
  foodList = newFood.asList()

  # Pacman position
  pacPos = currentGameState.getPacmanPosition()
  # ghostsState lists and ghost positions
  newGhostStates = currentGameState.getGhostStates()
  newGhostPos = [gs.getPosition() for gs in newGhostStates if gs.scaredTimer == 0]

  # find the total distance to food, then add one to avoid dividisoin by zero
  dFood = findTotalDistance(pacPos, foodList) + 1

  # get the size of the board
  boardSize = newFood.width + newFood.height

  # find the minimum distance to a ghost
  mdGhost = findMinDistance(pacPos, newGhostPos)
  
  # find the current score
  score = currentGameState.getScore()
  #bp = 100*math.atan(score)

  # if the next position puts you right next to a ghost then
  # don't take it becaues the ghost will advance to you AND FREAKING MURDER YOU
  if mdGhost == 1:
    mdGhost = -1*float("inf")

  # if the minimum ghost distance is less than 0 (hence there are no ghosts) then treat
  # it as if the ghost is as far away as possible and aim for food
  if mdGhost < 0:
    mdGhost = boardSize

  return (mdGhost/boardSize)*(1.0/dFood) + score #*bp

def findTotalDistance(origin, targets):
  """
  finds the total distance from the origin to every other target
  """
  total = 0
  (oX, oY) = origin
  # adds the change in x and y to the total distance
  for t in targets:
    (tX, tY) = t
    distance = abs(tX - oX) + abs(tY - oY)
    total += distance

  # return the total distance
  return total 

def findMinDistance(origin, targets):
  """
  finds the minimum distance between origin and all the targets
  """
  # return -1 if there are no targets 
  if len(targets) == 0:
    return -1

  # fence post problem
  (oX, oY) = origin
  (tX, tY) = targets[0]
  minimum = abs(tX - oX) + abs(tY - oY)
  
  # loops over all the targets
  for t in targets:
    (tX, tY) = t
    distance = abs(tX - oX) + abs(tY - oY)
    # potentially change min
    if distance < minimum:
      minimum = distance

  # return the total distance
  return minimum

# Abbreviation
better = betterEvaluationFunction

