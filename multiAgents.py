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
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
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
        foodList = newFood.asList()
        foodPossible = float('inf')
        negativeInfinity = -float('inf')

        for food in foodList:
            foodPossible = min(foodPossible, manhattanDistance(newPos, food))
        
        # Return small value if the ghost is near Pacman
        for agent in successorGameState.getGhostPositions():
            if (manhattanDistance(newPos, agent) < 1): 
                return negativeInfinity

        total = successorGameState.getScore() + 1.0 / foodPossible
        return total


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

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        # util.raiseNotDefined()
        def getMinValue(state, agentIndex, treeDepth):
            legalActions = state.getLegalActions(agentIndex)

            if legalActions == []: 
                return self.evaluationFunction(state)
            
            # Determine ghost and pacman mode
            if agentIndex != state.getNumAgents() - 1: 
                return min(getMinValue(state.generateSuccessor(agentIndex, i), agentIndex+1, treeDepth) for i in legalActions)
            else:
                return min(getMaxValue(state.generateSuccessor(agentIndex, i), treeDepth) for i in legalActions)
        
        # Reach last depth here
        def getMaxValue(state, treeDepth):
            legalActions = state.getLegalActions(0)
            if legalActions == [] or treeDepth == self.depth:
                return self.evaluationFunction(state)

            return max(getMinValue(state.generateSuccessor(0, i), 1, treeDepth + 1) for i in legalActions)
            
        actions = gameState.getLegalActions(0)
        path = max(actions, key=lambda i: getMinValue(gameState.generateSuccessor(0, i), 1, 1))
        return path

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"

        # util.raiseNotDefined()
        def getMinValue(state, agentIndex, treeDepth, x, y):
            legalActions = state.getLegalActions(agentIndex)
            if legalActions == []:
                return self.evaluationFunction(state)

            z = float('inf')
            for i in legalActions:
                newState = state.generateSuccessor(agentIndex, i)

                # Check to see whether it's the last ghost
                if agentIndex == state.getNumAgents() - 1:
                    newMove = getMaxValue(newState, treeDepth, x, y)
                else:
                    newMove = getMinValue(newState, agentIndex + 1, treeDepth, x, y)

                z = min(z, newMove)
                if z < x:
                    return z
                y = min(y, z)
            return z

        def getMaxValue(state, treeDepth, x, y):
            legalActions = state.getLegalActions(0)
            if legalActions == [] or treeDepth == self.depth:
                return self.evaluationFunction(state)

            z = -float('inf')
            if treeDepth == 0:
                bestMove = legalActions[0]
            for i in legalActions:
                newState = state.generateSuccessor(0, i)
                newMove = getMinValue(newState, 1, treeDepth + 1, x, y)
                if newMove > z:
                    z = newMove
                    if treeDepth == 0:
                        bestMove = i
                if z > y:
                    return z
                x = max(x, z)

            if treeDepth == 0:
                return bestMove
            return z

        bestMove = getMaxValue(gameState, 0, -float('inf'), float('inf'))
        return bestMove

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
        # util.raiseNotDefined()
        def getMaxValue(state, treeDepth):
            legalActions = state.getLegalActions(0)
            if legalActions == [] or treeDepth == self.depth:
                return self.evaluationFunction(state)

            v = max(expected(state.generateSuccessor(0, i), 1, treeDepth + 1) for i in legalActions)
            return v

        def expected(state, agentIndex, treeDepth):
            legalActions = state.getLegalActions(agentIndex)
            if legalActions == []:
                return self.evaluationFunction(state)

            pr = 1.0 / len(legalActions)
            z = 0
            for i in legalActions:
                newState = state.generateSuccessor(agentIndex, i)
                if agentIndex == state.getNumAgents() - 1:
                    z += getMaxValue(newState, treeDepth) * pr
                else:
                    z += expected(newState, agentIndex + 1, treeDepth) * pr
            return z

        legalActions = gameState.getLegalActions()
        bestMove = max(legalActions, key=lambda i: expected(gameState.generateSuccessor(0, i), 1, 1))
        return bestMove

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    # In some cases its better to not move when a dot may be besides because not doing any action is the same as
    # with other actions. That's why choosen several actions have same evaluation
    "*** YOUR CODE HERE ***"
    # util.raiseNotDefined()
    # Infinity = float('inf')
    # if currentGameState.isLose(): 
    #     return -Infinity
    # if currentGameState.isWin():  
    #     return Infinity
        
    # foods = currentGameState.getFood()
    # foodList = foods.asList()
    # stateGhost = currentGameState.getGhostStates()
    # pacman = currentGameState.getPacmanPosition()

    # closestFood = min(manhattanDistance(i, pacman) for i in foodList)
    # eat = sum([(manhattanDistance(i.getPosition(), pacman) < 3) for i in stateGhost])
    # notEat = sum([(i.scaredTimer == 0) for i in stateGhost])

    # return currentGameState.getScore() + 1.0 / closestFood + 1.0 *eat + 1.0 / (notEat + 0.5)

    # Infinity = float('inf')
    # if currentGameState.isLose(): 
    #     return -Infinity
    # if currentGameState.isWin():  
    #     return Infinity
        
    foods = currentGameState.getFood()
    foodList = foods.asList()
    # stateGhost = currentGameState.getGhostStates()
    pacman = currentGameState.getPacmanPosition()

    closestFood = min(manhattanDistance(pacman, i) for i in foodList) if foodList else 0.5
    return 1.0 / closestFood + currentGameState.getScore()

    # position = currentGameState.getPacmanPosition() == pacman
    # foods = currentGameState.getFood().asList() = foodList
    # closestFoodDis = min(manhattanDistance(position, food) for food in foods) if foods else 0.5
    # score = currentGameState.getScore()

    # '''
    #   Sometimes pacman will stay put even when there's a dot right besides, because 
    #   stop action has the same priority with other actions, so might be chosen when
    #   multiple actions have the same evaluation, upon which we can improve maybe.
    # '''
    # evaluation = 1.0 / closestFoodDis + score
    # return evaluation

# Abbreviation
better = betterEvaluationFunction
