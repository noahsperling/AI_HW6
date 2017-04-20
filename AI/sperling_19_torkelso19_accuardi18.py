# -*- coding: latin-1 -*-
import random
import sys

sys.path.append("..")  # so other modules can be found in parent dir
from Player import *
from Constants import *
from Construction import CONSTR_STATS
from Ant import UNIT_STATS
from Move import Move
from GameState import addCoords
from AIPlayerUtils import *


##
# AIPlayer
# Description: The responsibility of this class is to interact with the game by
# deciding a valid move based on a given game state. This class has methods that
# will be implemented by students in Dr. Nuxoll's AI course.
#
# Variables:
#   playerId - The id of the player.
##
class AIPlayer(Player):

    # maximum depth
    max_depth = 3

    # this AI's playerID
    me = -1

    # whether or not the playerID has been set up yet
    me_set_up = False

    # a list for consolidated states and their values
    state_value_list = []

    # __init__
    # Description: Creates a new Player
    #
    # Parameters:
    #   inputPlayerId - The id to give the new player (int)
    ##
    def __init__(self, inputPlayerId):
        super(AIPlayer, self).__init__(inputPlayerId, "NoahEricAi")
        # the coordinates of the agent's food and tunnel will be stored in these
        # variables (see getMove() below)
        self.myFood = None
        self.myTunnel = None

    ##
    # getPlacement
    #
    # The agent uses a hardcoded arrangement for phase 1 to provide maximum
    # protection to the queen.  Enemy food is placed randomly.
    #
    def getPlacement(self, currentState):
        self.myFood = None
        self.myTunnel = None
        if currentState.phase == SETUP_PHASE_1:
            return [(2, 1), (7, 1), #anthill, tunnel
                    #grass
                    (0, 3), (1, 3), (2, 3), (3, 3),
                    (4, 3), (5, 3), (6, 3),
                    (7, 3), (8, 3)]
        elif currentState.phase == SETUP_PHASE_2:
            #food placement
            #numToPlace = 2
            moves = []
            #for i in range(0, numToPlace):
            move = None

            #spots on the board
            sequence = [(0,6),(9,6),(1,6),(8,6),(2,6),(7,6),(3,6),(6,6),(4,6),(5,6),
                        (0,7),(9,7),(1,7),(8,7),(2,7),(7,7),(3,7),(6,7),(4,7),(5,7),
                        (0,8),(9,8),(1,8),(8,8),(2,8),(7,8),(3,8),(6,8),(4,8),(5,8),
                        (0,9),(9,9),(1,9),(8,9),(2,9),(7,9),(3,9),(6,9),(4,9),(5,9)
                        ]

            #finds valid locations
            sequence1 = [s for s in sequence if currentState.board[s[0]][s[1]].constr == None]

            anthillcoords = (0,0)
            tunnelcoords = (0,0)
            for x in range(0,10):
                for y in range(6,10):
                    if currentState.board[x][y].constr != None:
                        if currentState.board[x][y].constr.type == ANTHILL:
                            anthillcoords = (x,y)
                        if currentState.board[x][y].constr.type == TUNNEL:
                            tunnelcoords = (x,y)


            farthest = 0
            farthestCoords = (0,0)
            for s in sequence1:
                toAntHill = abs(approxDist(s, anthillcoords))
                toTunnel = abs(approxDist(s, tunnelcoords))
                if toAntHill >= toTunnel:
                    if(toTunnel >= farthest):
                        farthest = toTunnel
                        farthestCoords = s
                else:
                    if(toAntHill >= farthest):
                        farthest = toAntHill
                        farthestCoords = s

            moves.append(farthestCoords)

            sequence2 = [s for s in sequence1 if s[0] != farthestCoords[0] and s[1] != farthestCoords[1]]

            farthest = 0
            farthestCoords = (0,0)
            for s in sequence2:
                toAntHill = abs(approxDist(s,anthillcoords))
                toTunnel = abs(approxDist(s, tunnelcoords))
                if toAntHill >= toTunnel:
                    if(toTunnel > farthest):
                        farthest = toTunnel
                        farthestCoords = s
                else:
                    if(toAntHill > farthest):
                        farthest = toAntHill
                        farthestCoords = s

            moves.append(farthestCoords)
            return moves
        else:
            return None  # should never happen

            ##

    # move_search - recursive
    # Description: uses Minimax with alpha beta pruning to search for best next move
    #
    # Parameters:
    #   game_state - current state
    #   curr_depth - current search depth
    #   alpha      - the parent node's alpha value
    #   beta       - the parent node's beta value
    #
    # Return
    #   returns a move object
    ##
    def move_search(self, game_state, curr_depth, alpha, beta):

        # if max depth surpassed, return state evaluation
        if curr_depth == self.max_depth + 1:
            return self.evaluate_state(game_state)

        # list all legal moves
        move_list = listAllLegalMoves(game_state)

        # remove end turn move if the list isn't empty
        if not len(move_list) == 1:
            move_list.pop()

        # list of nodes, which contain the state, move, and eval
        node_list = []

        # generate states based on moves, evaluate them and put them into a list in node_list
        for move in move_list:
            state = getNextStateAdversarial(game_state, move)
            state_eval = self.evaluate_state(state)
            if not state_eval == 0.00001:
                node_list.append([state, move, state_eval])

        # sorts nodes in ascending order
        self.mergeSort(node_list)

        # reverses the order
        if not self.me == game_state.whoseTurn:
            move_list.reverse()

        best_nodes = []

        for i in range(0, 2):  # temporary
            if not len(node_list) == 0:
                best_nodes.append(node_list.pop())

        # best_val = -1

        # if not at the max depth, expand all the nodes in node_list and return
        # also pruned unnecessary nodes
        if curr_depth <= self.max_depth:
            for node in best_nodes:
                score = self.move_search(node[0], curr_depth + 1, alpha, beta)
                if game_state.whoseTurn == self.me:
                    if score > alpha:
                        alpha = score
                    if alpha >= beta:
                        break
                else:
                    if score < beta:
                        beta = score
                    if alpha >= beta:
                        break

        # if returns alpha or beta values depending on whoseTurn in the state, or the best move if at depth 0
        if game_state.whoseTurn == self.me and not curr_depth == 0:
            return alpha
        elif not game_state == self.me and not curr_depth == 0:
            return beta
        else:
            best_eval = -1
            best_node = []

            for node in best_nodes:
                if node[2] > best_eval:
                    best_eval = node[2]
                    best_node = node

            if not best_node == []:
                return best_node[1]
            else:
                return None

    ##
    # get_closest_enemy_dist - helper function
    #
    # returns distance to closest enemy from an ant
    ##
    def get_closest_enemy_dist(self, my_ant_coords, enemy_ants):
        closest_dist = 100
        for ant in enemy_ants:
            if not ant.type == WORKER:
                dist = approxDist(my_ant_coords, ant.coords)
                if dist < closest_dist:
                    closest_dist = dist
        return closest_dist

    ##
    # get_closest_enemy_worker_dist - helper function
    #
    # returns distance to closest enemy worker ant
    ##
    def get_closest_enemy_worker_dist(self, my_ant_coords, enemy_ants):
        closest_dist = 100
        for ant in enemy_ants:
            if ant.type == WORKER:
                dist = approxDist(my_ant_coords, ant.coords)
                if dist < closest_dist:
                    closest_dist = dist
        return closest_dist

    ##
    # get_closest_enemy_food_dist - helper function
    #
    # returns distance to closest enemy food
    ##
    def get_closest_enemy_food_dist(self, my_ant_coords, enemy_food_coords):

        enemy_food1_dist = approxDist(my_ant_coords, enemy_food_coords[0])
        enemy_food2_dist = approxDist(my_ant_coords, enemy_food_coords[1])

        if enemy_food1_dist < enemy_food2_dist:
            return enemy_food1_dist
        else:
            return enemy_food2_dist

    ##
    # evaluate_state
    #
    # Evaluates and scores a GameState Object
    #
    # Parameters
    #   state - the GameState object to evaluate
    #
    # Return
    #   a double between 0 and 1 inclusive
    ##
    def evaluate_state(self, state):
        # The AI's player ID
        me = state.whoseTurn
        # The opponent's ID
        enemy = (state.whoseTurn + 1) % 2

        # Get a reference to the player's inventory
        my_inv = state.inventories[me]
        # Get a reference to the enemy player's inventory
        enemy_inv = state.inventories[enemy]

        # Gets both the player's queens
        my_queen = getAntList(state, me, (QUEEN,))
        enemy_queen = getAntList(state, enemy, (QUEEN,))

        # Sees if winning or loosing conditions are already met
        if (my_inv.foodCount == 11) or (enemy_queen is None):
            return 1.0
        if (enemy_inv.foodCount == 11) or (my_queen is None):
            return 0.0

        # the starting value, not winning or losing
        eval = 0.0

        # important number
        worker_count = 0
        drone_count = 0

        food_coords = []
        enemy_food_coords = []

        foods = getConstrList(state, None, (FOOD,))

        # Gets a list of all of the food coords
        for food in foods:
            if food.coords[1] < 5:
                food_coords.append(food.coords)
            else:
                enemy_food_coords.append(food.coords)

        # coordinates of this AI's tunnel
        tunnel = my_inv.getTunnels()
        t_coords = tunnel[0].coords

        # coordinates of this AI's anthill
        ah_coords = my_inv.getAnthill().coords

        # A list that stores the evaluations of each worker
        wEval = []

        # A list that stores the evaluations of each drone, if they exist
        dEval = []

        # queen evaluation
        qEval = 0

        # iterates through ants and scores positioning
        for ant in my_inv.ants:

            # scores queen
            if ant.type == QUEEN:

                qEval = 50.0

                # if queen is on anthill, tunnel, or food it's bad
                if ant.coords == ah_coords or ant.coords == t_coords or ant.coords == food_coords[
                    0] or ant.coords == food_coords[1]:
                    qEval -= 10

                # if queen is out of rows 0 or 1 it's bad
                if ant.coords[0] > 1:
                    qEval -= 10

                # the father from enemy ants, the better
                qEval -= 2 * self.get_closest_enemy_dist(ant.coords, enemy_inv.ants)

            # scores worker to incentivize food gathering
            elif ant.type == WORKER:

                # tallies up workers
                worker_count += 1

                # if carrying, the closer to the anthill or tunnel, the better
                if ant.carrying:

                    wEval.append(100.0)

                    # distance to anthill
                    ah_dist = approxDist(ant.coords, ah_coords)

                    # distance to tunnel
                    t_dist = approxDist(ant.coords, t_coords)

                    # finds closest and scores
                    if t_dist < ah_dist:
                        wEval[worker_count - 1] -= 5 * t_dist
                    else:
                        wEval[worker_count - 1] -= 5 * ah_dist

                # if not carrying, the closer to food, the better
                else:

                    wEval.append(80.0)

                    # distance to foods
                    f1_dist = approxDist(ant.coords, food_coords[0])
                    f2_dist = approxDist(ant.coords, food_coords[1])

                    # finds closest and scores

                    if f1_dist < f2_dist:
                        wEval[worker_count - 1] -= 5 * f1_dist
                    else:
                        wEval[worker_count - 1] -= 5 * f2_dist

                        # the father from enemy ants, the better
                        # eval += -5 + self.get_closest_enemy_dist(ant.coords, enemy_inv.ants)

            # scores soldiers to incentivize the disruption of the enemy economy
            else:

                # tallies up soldiers
                drone_count += 1

                dEval.append(50.0)

                nearest_enemy_worker_dist = self.get_closest_enemy_worker_dist(ant.coords, enemy_inv.ants)

                # if there is an enemy worker
                if not nearest_enemy_worker_dist == 100:
                    dEval[drone_count - 1] -= 5 * nearest_enemy_worker_dist

                # if there isn't an enemy worker, go to the food
                else:
                    dEval[drone_count - 1] -= 5 * self.get_closest_enemy_food_dist(ant.coords, enemy_food_coords)

        # scores other important things

        # state eval
        sEval = 0

        # assesses worker inventory
        if worker_count == 2:
            sEval += 50
        elif worker_count < 2:
            sEval -= 10
        elif worker_count > 2:
            eval_num = 0.00001
            return eval_num

        # assesses drone inventory
        if drone_count == 2:
            sEval += 50
        elif drone_count < 2:
            sEval -= 10
        elif drone_count > 2:
            eval_num = 0.00001
            return eval_num

        # assesses food
        if not sEval == 0:
            sEval += 20 * my_inv.foodCount

        # a temporary variable
        temp = 0

        # scores workers
        for val in wEval:
            temp += val
        if worker_count == 0:
            wEvalAv = 0
        else:
            wEvalAv = temp / worker_count

        temp = 0

        # scores drones
        for val in dEval:
            temp += val

        if not drone_count == 0:
            dEvalAv = temp / drone_count
        else:
            dEvalAv = 0

        # total possible score
        total_possible = 100.0 + 50.0 + 50.0 + 300.0

        # scores total evaluation and returns
        eval = (qEval + wEvalAv + dEvalAv + sEval) / total_possible
        if eval <= 0:
            eval = 0.00002

        return eval


    ##
    # getAttack
    #
    # This agent never attacks
    #
    def getAttack(self, currentState, attackingAnt, enemyLocations):
        return enemyLocations[0]  # don't care


    ##
    # registerWin
    #
    # This agent doens't learn
    #
    def registerWin(self, hasWon):
        # method templaste, not implemented
        pass


    ##
    # merge_sort
    #
    # useful for sorting the move list from least to greatest in nlog(n) time
    ##
    def mergeSort(self, alist):
        if len(alist) > 1:
            mid = len(alist) // 2
            lefthalf = alist[:mid]
            righthalf = alist[mid:]

            self.mergeSort(lefthalf)
            self.mergeSort(righthalf)

            i = 0
            j = 0
            k = 0
            while i < len(lefthalf) and j < len(righthalf):
                if lefthalf[i][2] < righthalf[j][2]:
                    alist[k] = lefthalf[i]
                    i = i + 1
                else:
                    alist[k] = righthalf[j]
                    j = j + 1
                k = k + 1

            while i < len(lefthalf):
                alist[k] = lefthalf[i]
                i = i + 1
                k = k + 1

            while j < len(righthalf):
                alist[k] = righthalf[j]
                j = j + 1
                k = k + 1


    ##
    # consolidate_state
    #
    # returns a simplified version of a GameState object
    #
    # Parameters
    #   state - the state to simplify
    #
    # Return
    #   simplified_state - the simplified version of the state
    ##
    def consolidate_state(self, state):

        simplified_state = state

        return simplified_state

    ##
    #
    #
    #
    #
    #
    #
    #
    ##
    def append_state_to_list(self, state, utility):
        self.state_value_list.append([state, utility])
        return


    ##
    # write_state_list_to_file
    #
    #
    #
    ##
    def write_state_list_to_file(self, filename):
        with open(filename, "w") as f:
            for state in self.state_value_list:
                for num in state[0]:
                    f.write("%lf," % num)
                f.write("%lf\n" % state[1])


    ##
    # read_states_from_file
    #
    # Parameters
    #   filename - file to open
    #
    # Return
    #   returns nothing
    #
    ##
    def read_states_from_file(self, filename):
        with open(filename, "r") as f:
            states = f.splitlines()
            for s in states:
                nums = s.split(",")
                length = len(s)
                state = s[:length - 1]
                eval = s[length - 1]
                self.state_value_list.append([state, eval])






