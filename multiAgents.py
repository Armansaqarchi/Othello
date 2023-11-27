from Agents import Agent
import util
import random
import sys
from Game import GameState

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """
    def __init__(self, *args, **kwargs) -> None:
        self.index = 0 # your agent always has index 0

    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        It takes a GameState and returns a tuple representing a position on the game board.
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions(self.index)

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        The evaluation function takes in the current and proposed successor
        GameStates (Game.py) and returns a number, where higher numbers are better.
        You can try and change this evaluation function if you want but it is not necessary.
        """
        nextGameState = currentGameState.generateSuccessor(self.index, action)
        return nextGameState.getScore(self.index) - currentGameState.getScore(self.index)


def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    Every player's score is the number of pieces they have placed on the board.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore(0)


class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxAgent, AlphaBetaAgent & ExpectimaxAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (Agents.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2', **kwargs):
        self.index = 0 # your agent always has index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)


class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent which extends MultiAgentSearchAgent and is supposed to be implementing a minimax tree with a certain depth.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(**kwargs)

    def getAction(self, state : GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction

        But before getting your hands dirty, look at these functions:

        gameState.isGameFinished() -> bool
        gameState.getNumAgents() -> int
        gameState.generateSuccessor(agentIndex, action) -> GameState
        gameState.getLegalActions(agentIndex) -> list
        self.evaluationFunction(gameState) -> float
        """
        agents : int = state.getNumAgents()
        best_action : tuple[int, int]
        best_profit : int = 1 - sys.maxsize
        for legal_action in state.getLegalActions(index=0):
            prev_profit = best_profit
            new_p = self._min_max_policy(index=1, state=state, depth=2)
            best_profit = max(best_profit, new_p)
            if best_profit > prev_profit:
                best_action = legal_action
        return best_action

    
    def _min_max_policy(self, index : int, state : GameState, depth) ->  int:
        if state.isGameFinished():
            return state.getScore(index = index)
        elif (depth == 0):
            return scoreEvaluationFunction(state)
        profit : int
        if index == 0:
            func = max
            profit = 1 - sys.maxsize
        else:
            func = min
            profit = sys.maxsize
        for legal_action in state.getLegalActions(index=index):
            next_state = state.generateSuccessor(index, action=legal_action)
            agents = state.getNumAgents()
            choice = self._min_max_policy((index+1)%(agents), next_state, depth-1)
            prev_profit : int = profit
            profit = func(profit, choice)
        return profit

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning. It is very similar to the MinimaxAgent but you need to implement the alpha-beta pruning algorithm too.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(**kwargs)

    def getAction(self, state : GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction

        You should keep track of alpha and beta in each node to be able to implement alpha-beta pruning.
        """
        best_action : tuple[int, int]
        best_score : int = 1 - sys.maxsize
        a : int = 1 - sys.maxsize
        b : int = sys.maxsize

        for action in state.getLegalActions(index=0):
            next_state : GameState = state.generateSuccessor(agentIndex=0, action=action)
            new_score = self._alpha_beta_policy(next_state, 1, a, b, 4)
            if new_score > best_score:
                best_score = new_score
                best_action = action
        return best_action
    
    def _alpha_beta_policy(self, state : GameState, index : int, a : int, b : int, depth : int) -> int:
        """
        returns the best option for the current state depending on its type
        but most of the time, the exact value is not returned due to pruning
        the reason behind the scenes for this is that no matter how exact the value becomes,
        the value is not an option for parents of this state
        """
        if state.isGameFinished():
            return state.getScore(index = 0)
        elif depth == 0:
            return scoreEvaluationFunction(state)
        if index == 0:
            v : int = 1 - sys.maxsize
        else:
            v : int = sys.maxsize

        for legal_action in state.getLegalActions(index = index):
            next_state = state.generateSuccessor(agentIndex=index, action=legal_action)
            next_score : int = self._alpha_beta_policy(next_state, (index+1)%state.getNumAgents(), a, b, depth-1)
            
            if index == 0:
                v = max(next_score, v)
                if v > b:
                    return v
                a = max(a, v)
            else:
                v = min(next_score, v)
                if v < a:
                    return v
                b = min(b, v)
        return v
        

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent which has a max node for your agent but every other node is a chance node.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(**kwargs)

    def getAction(self, state : GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All opponents should be modeled as choosing uniformly at random from their
        legal moves.
        """
        max_value : int = 1 - sys.maxsize
        max_case : tuple[int, int]
        index = 0
        for action in state.getLegalActions(index = 0):   
            next_state = state.generateSuccessor(agentIndex=index, action=action)
            value = self.expectimax_policy(next_state, index=index+1, depth=3)
            if value > max_value:
                max_case = action
                max_value = value
        return max_case


    def max_policy(self, score : int, value : int):
        return max(score, value)
    
    def expect_policy (self, score : int, value : int):
        return score + value

    def expectimax_policy(self, state : GameState, index : int, depth : int):
        if state.isGameFinished():
            return state.getScore(index=0)
        elif depth == 0:
            return scoreEvaluationFunction(state)
        best_score : int = 1 - sys.maxsize
        if index == 0:
            func = self.max_policy
        else:
            best_score = 0
            func = self.expect_policy

        legal_actions = state.getLegalActions(index=0)
        num_of_actions = len(legal_actions)
        for action in legal_actions:
            next_state = state.generateSuccessor(index, action=action)
            value = self.expectimax_policy(next_state, (index+1)%state.getNumAgents(), depth-1)
            best_score = func(best_score, value=value)
        if index != 0:
            best_score /= num_of_actions

        return best_score



def betterEvaluationFunction(currentGameState):
    """
    Your extreme evaluation function.

    You are asked to read the following paper on othello heuristics and extend it for two to four player rollit game.
    Implementing a good stability heuristic has extra points.
    Any other brilliant ideas are also accepted. Just try and be original.

    The paper: Sannidhanam, Vaishnavi, and Muthukaruppan Annamalai. "An analysis of heuristics in othello." (2015).

    Here are also some functions you will need to use:
    
    gameState.getPieces(index) -> list
    gameState.getCorners() -> 4-tuple
    gameState.getScore() -> list
    gameState.getScore(index) -> int

    """
    
    "*** YOUR CODE HERE ***"

    # parity

    # corners

    # mobility

    # stability
    
    util.raiseNotDefined()

def mobility(self, currentGameState : GameState):
    min_mobility : int = sys.maxsize
    max_mobility : int = 1 - sys.maxsize
    agents : int = currentGameState.getNumAgents()
    for idx in range(0, agents):
        value : int = len(currentGameState.getLegalActions(index=idx))
        if value > max_mobility:
            max_mobility = value
            continue
        elif value < min_mobility:
            min_mobility = value
    if max_mobility + min_mobility == 0:
        return 0
    
    return 100 * (max_mobility - min_mobility) / (max_mobility + min_mobility)


def parity(self, currentGameState : GameState):
    min_coins: int = sys.maxsize
    max_coins : int = 1 - sys.maxsize
    agents : int = currentGameState.getNumAgents()
    for idx in range(0, agents):
        value : int = len(currentGameState.getPieces(index=idx))
        if value > max_coins:
            max_coins = value
            continue
        elif value < min_coins:
            min_coins = value
    return 100 * (max_coins - min_coins) / (max_coins + min_coins)
    

def stablity(self, currentGameState : GameState):
    ...

def corners(self, currentGameState : GameState):
    min_corner : int = sys.maxsize
    max_corner : int = 1 - sys.maxsize
    agents : int = currentGameState.getNumAgents()
    corners_list = [0] * agents
    for i in currentGameState.getCorners():
        if i == -1:
            continue
        corners_list[i] += 1
    for idx in range(0, agents):
        value = corners_list[idx]    
        if value > max_corner:
            max_corner = value
            continue
        elif value < min_corner:
            min_corner = value
    return 100 * (max_corner - min_corner) / (max_corner + min_corner)

# Abbreviation
better = betterEvaluationFunction