# search.py
# ---------
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


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

from game import Directions
import util

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def generalTreeSearch(problem, structure):
    """
    a generalization that accepts a structure and performes a DPF or BFS with it
    """

    # for the off case when you start at the end line
    if problem.isGoalState(problem.getStartState()):
        return ['Stop']

    # a list of all explored locations
    explored = []
    # we are going to store the entire tuple in the structure, so it can be later
    # used as a path
    # get all the successors to populate the structure with them
    startSuccessors = problem.getSuccessors(problem.getStartState())
    for s in startSuccessors:
        structure.push([s])

    # keep looking for the goal while you can
    while not structure.isEmpty():
        # pop the state at the top of the structure
        path = structure.pop() # a list of tuples
        currentState = path[len(path) - 1] # one tuple
        (location, direction, cost) = currentState # the elements of this tuple
        # check if the current state is the goal state
        if problem.isGoalState(location):
            # create the path then return it
            answerPath = []
            for i in range(0, len(path)):
                (cLocation, cDirection, cCost) = path[i]
                answerPath.append(cDirection)
            return answerPath
        # proceed only if this location have not been explored before
        elif location not in explored:
            # add the current state to explored
            explored.append(location)
            # get all the succesors of this state as tuples
            successors = problem.getSuccessors(location)
            # create the paths of the successors and push them in
            for t in successors:
                # copy the current path
                tuplePath = list(path) #DOES THIS REALLY DEEP COPY OR NOT
                # append this tuple to the path after its parent
                tuplePath.append(t)
                # add the new path to the structure
                structure.push(tuplePath)
        # the the state is explored then ignore it
    
    # if the structure is empty then we failed to find the goal state, return None
    print "the structure is empty"
    return None

def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print "Start:", problem.getStartState()
    print "Is the start a goal?", problem.isGoalState(problem.getStartState())
    print "Start's successors:", problem.getSuccessors(problem.getStartState())
    """
    "*** YOUR CODE HERE ***"
    structure = util.Stack()
    return generalTreeSearch(problem, structure)

def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    structure = util.Queue()
    return generalTreeSearch(problem, structure)

def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    # create the function to calculate the priority
    def function(path):
        actions = []
        for s in path: 
            actions.append(s[1])
        cost = problem.getCostOfActions(actions)
        return cost

    # priority queue with the lambda as the priority function
    structure = util.PriorityQueueWithFunction(function)
    return generalTreeSearch(problem, structure)

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"

    # function to calculate the 
    def function(path):
        actions = []
        # get the sequence of actions in the path
        for s in path: 
            actions.append(s[1])
        # extract the last position in the path to perform the heuristic on it
        (lLocation, lDirection, lCost) = path[len(path) - 1]
        # cost = f(n) = g(n) + h(n)
        cost = problem.getCostOfActions(actions) + heuristic(lLocation, problem)
        return cost

    # priority queue with the lambda as the priority function
    structure = util.PriorityQueueWithFunction(function)
    return generalTreeSearch(problem, structure)


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
