# search.py
# ---------
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


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util
from game import Directions



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

    #data stuctures
    parent = {}
    path = [] 
    visited = set()
    stackState = util.Stack()
    #push start state
    state = problem.getStartState()
    stackState.push((state,))
   
    while not stackState.isEmpty():	
    	state = stackState.pop()
    	visited.update([state[0]])
        #if end state break
        if problem.isGoalState(state[0]):
            break
    	nextStates = problem.getSuccessors(state[0])
    	for nextState in nextStates:     
            if nextState[0] not in visited:              
                stackState.push(nextState)
                parent[nextState] = state

    #construct the path
    path.append(state)
    while  problem.getStartState() != parent[state][0]:
        path.append(parent[state])
        state = path[-1]
    path.reverse()
    path = [i[1] for i in path]
    
    return path
    #util.raiseNotDefined()
def generalGraphSearch(problem, structure):
    """
    Defines a general algorithm to search a graph.
    Parameters are structure, which can be any data structure with .push() and .pop() methods, and problem, which is the
    search problem.
    """

    # Push the root node/start into the data structure in this format: [(state, action taken, cost)]
    # The list pushed into the structure for the second node will look something like this:
    # [(root_state, "Stop", 0), (new_state, "North", 1)]
    structure.push([(problem.getStartState(), "Stop", 0)])

    # Initialise the list of visited nodes to an empty list
    visited = []

    # While the structure is not empty, i.e. there are still elements to be searched,
    while not structure.isEmpty():
        # get the path returned by the data structure's .pop() method
        path = structure.pop()

        # The current state is the first element in the last tuple of the path
        # i.e. [(root_state, "Stop", 0), (new_state, "North", 1)][-1][0] = (new_state, "North", 1)[0] = new_state
        curr_state = path[-1][0]

        # if the current state is the goal state,
        if problem.isGoalState(curr_state):
            # return the actions to the goal state
            # which is the second element for each tuple in the path, ignoring the first "Stop"
            return [x[1] for x in path][1:]

        # if the current state has not been visited,
        if curr_state not in visited:
            # mark the current state as visited by appending to the visited list
            visited.append(curr_state)

            # for all the successors of the current state,
            for successor in problem.getSuccessors(curr_state):
                # successor[0] = (state, action, cost)[0] = state
                # if the successor's state is unvisited,
                if successor[0] not in visited:
                    # Copy the parent's path
                    successorPath = path[:]
                    # Set the path of the successor node to the parent's path + the successor node
                    successorPath.append(successor)
                    # Push the successor's path into the structure
                    structure.push(successorPath)

    # If search fails, return False
    return False

def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    #data stuctures


    parent = {}
    path = [] 
    visited = set()
    queued = set()
    stackState = util.Queue()
    #push start state
    state = problem.getStartState()
    stackState.push((state,))
    queued.update((state,))
    i = 0
    nee = []
    breakk = False
    
    while not stackState.isEmpty():
        i += 1	
        print i , stackState.list
        state = stackState.pop()
        visited.update([state[0]])

        #if end state break
        if problem.isGoalState(state[0]):
            break
        
        

        for nextState in problem.getSuccessors(state[0]):   
            if ((nextState[0] not in visited)):
                if problem.isGoalState(nextState[0]):
                    stackState.push(nextState)
                    parent[nextState] = state
                    state = nextState
                    breakk = True  
                else:          
                    stackState.push(nextState)
                    parent[nextState] = state

        if breakk:
            break

    #construct the path
    path.append(state)
    while  problem.getStartState() != parent[state][0]:
        path.append(parent[state])
        state = path[-1]
    path.reverse()
    path = [i[1] for i in path]

    print path
    return path
    """
    from util import Queue    
    Frontier=Queue()
    Explored_set = []

    pathlist = []
    Frontier.push((problem.getStartState(),pathlist))

    while Frontier:
    
        state,actions_made = Frontier.pop()
        Explored_set.append(state)

        if problem.isGoalState(state):
            return actions_made
        
        for succ in problem.getSuccessors(state): 
            if succ[0] not in Explored_set:
                Explored_set.append(succ[0])
                Frontier.push((succ[0],actions_made+[succ[1]]))  

    return []
    """
    #util.raiseNotDefined()
    #util.raiseNotDefined()

def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
