from Agents import Agent
import util
import random
import sys
from Game import GameState
import time

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
        depth = self.depth
        for legal_action in state.getLegalActions(index=0):
            new_p = self._min_max_policy(index=1, state=state, depth=depth-1)
            if new_p > best_profit:
                best_profit = new_p
                best_action = legal_action
        return best_action

    
    def _min_max_policy(self, index : int, state : GameState, depth) ->  int:
        if state.isGameFinished():
            return state.getScore(index = index)
        elif (depth == 0):
            return self.evaluationFunction(state)
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

            profit = func(profit, choice)
        return profit

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning. It is very similar to the MinimaxAgent but you need to implement the alpha-beta pruning algorithm too.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(**kwargs)

    def getAction(self, state):
        """
        Returns the minimax action using self.depth and self.evaluationFunction

        You should keep track of alpha and beta in each node to be able to implement alpha-beta pruning.
        """
        best_action : tuple[int, int]
        best_score : int = 1 - sys.maxsize
        a : int = 1 - sys.maxsize
        b : int = sys.maxsize

        for action in state.getLegalActions(index=0):
            next_state = state.generateSuccessor(agentIndex=0, action=action)
            new_score = self._alpha_beta_policy(next_state, 1, a, b, 4)
            if new_score > best_score:
                best_score = new_score
                best_action = action
        return best_action
    
    def _alpha_beta_policy(self, state, index : int, a : int, b : int, depth : int) -> int:
        """
        returns the best option for the current state depending on its type
        but most of the time, the exact value is not returned due to pruning
        the reason behind the scenes for this is that no matter how exact the value becomes,
        the value is not an option for parents of this state
        """
        if state.isGameFinished():
            return state.getScore(index = 0)
        elif depth == 0:
            return self.evaluationFunction(state)
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
    
    # class TestAlphaBeta:
    #     GRAPH = {
    #         "A1" : ["B1", "B2"],
    #         "B1" : ["C1", "C2"],
    #         "B2" : ["C3", "C4"],
    #         "C1" : ["D1", "D2"],
    #         "C2" : ["D3", "D4"],
    #         "C3" : ["D5", "D6"],
    #         "C4" : ["D7", "D8"],
    #         "D1" : ["A", "B"],
    #         "D2" : ["C", "D"],
    #         "D3" : ["E", "F"],
    #         "D4" : ["G", "H"],
    #         "D5" : ["I", "J"],
    #         "D6" : ["K", "L"],
    #         "D7" : ["M", "N"],
    #         "D8" : ["O", "P"],
    #     }
    #     TERMINAL_VALUES = {
    #         "A" : 6,
    #         "B" : 8,
    #         "C" : 10,
    #         "D" : 11,
    #         "E" : 13,
    #         "F" : 3,
    #         "G" : 5,
    #         "H" : 9,
    #         "I" : 4,
    #         "J" : 7,
    #         "K" : 1,
    #         "L" : 0,
    #         "M" : 12,
    #         "N" : 15,
    #         "O" : 2,
    #         "P" : 14
    #     }

    #     def __init__(self, name, a, b, isTerminal, score) -> None:
    #         self.name = name
    #         self.isTerminal = isTerminal
    #         self.a = a
    #         self.b = b
    #         if isTerminal:
    #             self.score = score

    #     def getLegalActions(self):
    #         return self.GRAPH[self.name]

    #     def getScore(self):
    #         if self.isTerminal:
    #             raise Exception("not a terminal state to perform score")
    #         return self.score

    #     def isGameFinished(self):
    #         return self.isTerminal
        
    #     def generateSuccessor(self, action : int):
    #         state = AlphaBetaAgent.TestAlphaBeta(
    #             a = self.a,
    #             b = self.b,
    #             isTerminal= False,
    #             score=False,
    #             name= self.GRAPH[self.name][action],
    #         )
    #         if self.name[0] == "D":
    #             state.nextTerminal = True
    #             state.score = self.TERMINAL_VALUES[state.name]

    # def test_alpha_beta(self):
    #     state = AlphaBetaAgent.TestAlphaBeta(
    #         name = "A1",
    #         a = 1 - sys.maxsize,
    #         b = sys.maxsize,
    #         isTerminal=False,
    #         score=None
    #     )
    #     self.getAction(state=state)
        
        

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
    gameState.getcorns() -> 4-tuple
    gameState.getScore() -> list
    gameState.getScore(index) -> int

    """
    
    "*** YOUR CODE HERE ***"

    par = parity(currentGameState=currentGameState)

    corns = corners(currentGameState=currentGameState)

    mob = mobility(currentGameState=currentGameState)

    stab = stablity(currentGameState=currentGameState)


    return 1 * corns + 10 * stab + 1 * mob + 1 * par
    

def mobility(currentGameState : GameState):
    max_player_mobility = len(currentGameState.getLegalActions(index=0))
    min_player_mobility = 0
    agents = currentGameState.getNumAgents()
    for i in range(1, agents):
        min_player_mobility += len(currentGameState.getLegalActions(index=i))
    if min_player_mobility + max_player_mobility == 0:
        return 0
    
    return (max_player_mobility - min_player_mobility) / (max_player_mobility + min_player_mobility)

def parity(currentGameState : GameState):
    players_coins = [0, 0]
    players_coins[0] = len(currentGameState.getPieces(index = 0))
    agents = currentGameState.getNumAgents()
    for i in range(1, agents):
        players_coins[1] += len(currentGameState.getPieces(index=i))
    if players_coins[1] + players_coins[0] == 0:
        return 0
    return (players_coins[0] - players_coins[1]) / (players_coins[0] + players_coins[1])


def stablity(currentGameState : GameState):
    players_stablity = [0, 0]
    static_weights = [
        [4,  -3,  2,  2,  2,  2, -3,  4],
        [-3, -4, -1, -1, -1, -1, -4, -3],
        [2,  -1,  1,  0,  0,  1, -1,  2],
        [2,  -1,  0,  1,  1,  0, -1,  2],
        [2,  -1,  0,  1,  1,  0, -1,  2],
        [2,  -1,  1,  0,  0,  1, -1,  2],
        [-3, -4, -1, -1, -1, -1, -4, -3],
        [4,  -3,  2,  2,  2,  2, -3,  4],
    ]

    for i in range (8):
        for j in range(8):
            index = currentGameState.data.board[i][j]
            if index == -1:
                continue
            if index == 0:
                players_stablity[0] += static_weights[i][j]
            else:
                players_stablity[1] += static_weights[i][j]

    if players_stablity[0] + players_stablity[1] == 0:
        return 0
    
    return 100 * (players_stablity[0] - players_stablity[1])/ (players_stablity[0] + players_stablity[1])

def corners(currentGameState : GameState):
    corners = [0, 0]
    for i in currentGameState.getCorners():
        if i == -1:
            continue
        if i == 0:
            corners[0] += 1
        else:
            corners[1] += 1

    if corners[0] + corners[1] == 0:
        return 0
    
    return 100 * (corners[0] - corners[1]) / (corners[0] + corners[1])

# Abbreviation
better = betterEvaluationFunction