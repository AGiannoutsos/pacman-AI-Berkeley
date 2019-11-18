# multiAgents.py
# --------------
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


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent
def ghostDistance(x):
  return (0.05*(x-20)**2) + 50
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
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]


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
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        # Get distance from pacman to ghosts in new position
        pacmanGhostDistance = [util.manhattanDistance(newPos,ghost) for ghost in successorGameState.getGhostPositions()]
        minGhostDis = min(pacmanGhostDistance)

        # Get closest food
        foodPos = []
        minFoodDist = 0
        for i, food_x in enumerate(newFood):
          for j, food in enumerate(newFood[i]):
            if food:
              foodPos.append(util.manhattanDistance(newPos,(i,j)))
        if foodPos:
          minFoodDist = min(foodPos)

        # Get capsule position
        minCapsuleDist = 0
        capsulesPos = [util.manhattanDistance(newPos,capsule) for capsule in successorGameState.getCapsules()]
        if capsulesPos:
          minCapsuleDist = min(capsulesPos)

        # Promote a position if has food
        if currentGameState.hasFood(newPos[0], newPos[1]):
          minFoodDist -= 10
        
        return  minGhostDis - minFoodDist

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
        # for i in range(4):
          # print  gameState.getLegalActions(i),10, self.evaluationFunction(gameState),23 ,self.depth

        def minimax(state, agent, depth):
          legalMoves = state.getLegalActions(agent)
          # If no moves or end of depth return
          if (not legalMoves) or (depth == self.depth):
            return (self.evaluationFunction(state), Directions.STOP)

          # Ghost agents that are not 0 or over the total agent number
          if (agent > 0 and agent < gameState.getNumAgents()):
            # For next Ghost agent if the next is pacman then update the depth
            if agent+1 == gameState.getNumAgents():
              agent = -1
              depth += 1

            costPerAction = []
            for action in legalMoves:
              costPerAction.append(minimax(state.generateSuccessor(agent, action),agent + 1,depth)[0])
            minCost = (min(costPerAction))
            return (minCost , legalMoves[costPerAction.index(minCost)])
            
          # Max player pacman
          else:
            costPerAction = []
            for action in legalMoves:
              costPerAction.append(minimax(state.generateSuccessor(0, action),1,depth)[0])
            maxCost = (max(costPerAction))
            return (maxCost , legalMoves[costPerAction.index(maxCost)])

        return minimax(gameState,0,0)[1]

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        inf = 1000000000
        

        def AlphaBeta(state, agent, depth, a, b):
          legalMoves = state.getLegalActions(agent)
          # If no moves or end of depth return
          if (not legalMoves) or (depth == self.depth):
            return (self.evaluationFunction(state), Directions.STOP)

          # Ghost agents that are not 0 or over the total agent number
          if (agent > 0 and agent < gameState.getNumAgents()):
            # For next Ghost agent if the next is pacman then update the depth
            if agent+1 == gameState.getNumAgents():
              agent = -1
              depth += 1

            costPerAction = []
            value = inf
            for action in legalMoves:
              value = AlphaBeta(state.generateSuccessor(agent, action),agent + 1,depth, a, b)[0]
              costPerAction.append(value)
              # if value lower than the lower bound then certainly ghost will chose that path
              if value < a:
                break
              b = min(value,b)
            minCost = min(costPerAction)
            return (minCost , legalMoves[costPerAction.index(minCost)])
            
          # Max player pacman
          else:
            costPerAction = []
            value = inf
            for action in legalMoves:
              value = AlphaBeta(state.generateSuccessor(0, action),1,depth, a, b)[0]
              costPerAction.append(value)
              # if value greater than upper bound then pacman certainly will chose that path
              if value > b:
                break
              a = max(value,a)
            maxCost = max(costPerAction)
            return (maxCost , legalMoves[costPerAction.index(maxCost)])

        return AlphaBeta(gameState,0,0,-inf,inf)[1]

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
        def minimax(state, agent, depth):
          legalMoves = state.getLegalActions(agent)
          # If no moves or end of depth return
          if (not legalMoves) or (depth == self.depth):
            return (self.evaluationFunction(state), Directions.STOP)

          # Chance nodes
          # Calculate the propability 
          if (agent > 0 and agent < gameState.getNumAgents()):
            # For next Ghost agent if the next is pacman then update the depth
            if agent+1 == gameState.getNumAgents():
              agent = -1
              depth += 1

            costPerAction = []
            for action in legalMoves:
              costPerAction.append(minimax(state.generateSuccessor(agent, action),agent + 1,depth)[0])
            # sum/number of moves because ghosts are doing uniformly random actions
            averageCost = sum(costPerAction)/len(legalMoves) 
            # actions dont matter in this occasion
            return (averageCost, Directions.STOP)
            
          # Max player pacman
          else:
            costPerAction = []
            for action in legalMoves:
              costPerAction.append(minimax(state.generateSuccessor(0, action),1,depth)[0])
            maxCost = (max(costPerAction))
            return (maxCost , legalMoves[costPerAction.index(maxCost)])

        return minimax(gameState,0,0)[1]


def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    pacmanPos = currentGameState.getPacmanPosition()
    foodPos = currentGameState.getFood().asList()
    ghostsPos = currentGameState.getGhostPositions()

    # Get distance from pacman to ghosts in new position
    pacmanGhostDist = [util.manhattanDistance(pacmanPos,ghost) for ghost in ghostsPos]
    minGhostDist = min(pacmanGhostDist)

    # Get closest food position
    minFoodDist = 0
    pacmanFoodDist = [util.manhattanDistance(pacmanPos,food) for food in foodPos]
    if pacmanFoodDist:
      minFoodDist = min(pacmanFoodDist)

    # Get number of close foods under 4 distance
    closeFoods = []
    if pacmanFoodDist:
      closeFoods = [food for food in foodPos if util.manhattanDistance(pacmanPos,food) < 4]

    
    return minGhostDist - minFoodDist + 0.8*currentGameState.getScore()

# Abbreviation
better = betterEvaluationFunction

