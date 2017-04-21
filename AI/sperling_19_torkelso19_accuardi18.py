# -*- coding: latin-1 -*-
import random
import sys
import math

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
    max_depth = 0

    # this AI's playerID
    me = -1

    # whether or not the playerID has been set up yet
    me_set_up = False


    #TD Learning Variables Below

    #FileName
    filepath = "C://Users/ejit5_000/PycharmProjects/AI_HW6/sperling19_torkelso19_accuardi18.txt"

    # a list for consolidated states and their values
    state_value_list = []

    #Discount Factor
    gamma = 0.8

    #Learning Rate
    alpha = 1.0

    #Eligibility Trace Stuff
    e_lambda = 1.0
    e_size = 50
    eligibility_queue = []

    #Game Counter
    numGames = 0

    # __init__
    # Description: Creates a new Player
    #
    # Parameters:
    #   inputPlayerId - The id to give the new player (int)
    ##
    def __init__(self, inputPlayerId):
        super(AIPlayer, self).__init__(inputPlayerId, "NoahErikNickAi")
        # the coordinates of the agent's food and tunnel will be stored in these
        # variables (see getMove() below)
        self.myFood = None
        self.myTunnel = None

        #Try to open utility file and if doesn't exist create it
        try:
            self.read_states_from_file(self.filepath)
        except IOError:
            self.write_state_list_to_file(self.filepath)

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
    # getMove
    # Description: Gets the next move from the Player.
    #
    # Parameters:
    #   currentState - The state of the current game waiting for the player's move (GameState)
    #
    # Return: The Move to be made
    ##
    def getMove(self, currentState):

        if not self.me_set_up:
            self.me = currentState.whoseTurn

        # searches for best move
        selectedMove = self.move_search(currentState, 0, -(float)("inf"), (float)("inf"))

        #TD Learning
        self.append_state(self.consolidate_state(currentState))
        index_of_state = self.state_value_list.index([self.consolidate_state(currentState), self.find_state_eval(self.consolidate_state(currentState))])

        self.eligibility_queue.insert(0,index_of_state)
        if len(self.eligibility_queue) > self.e_size:
            self.eligibility_queue.pop()

        #TD Learning Equation
        if len(self.eligibility_queue) > 1:
            self.e_lambda = 1
            for i in range(len(self.eligibility_queue)):
                if i == 0: continue
                else:
                    self.state_value_list[self.eligibility_queue[i]][1] = -0.001 + self.alpha*self.e_lambda*(-.001 + self.gamma*self.state_value_list[self.eligibility_queue[i-1]][1] - self.state_value_list[self.eligibility_queue[i]][1])
                    self.e_lambda *= 0.9

        # if not None, return move, if None, end turn
        if not selectedMove == None:
            return selectedMove
        else:
            return Move(END, None, None)


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
            evaluation = self.find_state_eval(game_state)
            if evaluation == -25655:
                evaluation = 0
            return evaluation

        # list all legal moves
        move_list = listAllLegalMoves(game_state)

        # remove end turn move if the list isn't empty
        #if not len(move_list) == 1:
        #    move_list.pop()

        # list of nodes, which contain the state, move, and eval
        node_list = []

        # generate states based on moves, evaluate them and put them into a list in node_list
        for move in move_list:
            state = getNextStateAdversarial(game_state, move)
            state_eval = self.find_state_eval(state)
            if state_eval == -25655:
                state_eval = 0
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

            random.shuffle(best_nodes)

            for node in best_nodes:
                if node[2] > best_eval:
                    best_eval = node[2]
                    best_node = node

            if not best_node == []:
                return best_node[1]
            else:
                return None


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
    # This agent doesn't learn
    #
    def registerWin(self, hasWon):
        self.numGames += 1

        reward = 0
        if hasWon:
            reward = 1
        else:
            reward = -1

        # TD Learning Equation
        if len(self.eligibility_queue) > 1:
            self.e_lambda = 1
            for i in range(len(self.eligibility_queue)):
                if i == 0:
                    self.state_value_list[self.eligibility_queue[i]][1] = -0.001 + self.alpha * self.e_lambda * (-.001 + self.gamma * reward - self.state_value_list[self.eligibility_queue[i]][1])
                else:
                    self.state_value_list[self.eligibility_queue[i]][1] = -0.001 + self.alpha * self.e_lambda * (
                    -.001 + self.gamma * self.state_value_list[self.eligibility_queue[i - 1]][1] -
                    self.state_value_list[self.eligibility_queue[i]][1])
                    self.e_lambda *= 0.9

        self.write_state_list_to_file(self.filepath)

        if self.alpha > 0.001:
            self.alpha = math.pow(1.1,-0.25*self.numGames)

        self.eligibility_queue = []

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

        # food inventory numbers
        player_food = 0

        # queen info
        queen_health = 0

        # worker info
        worker_count = 0
        w_distance_to_move = 0

        # drone info
        fighter_count = 0

        # The AI's player ID
        me = state.whoseTurn

        # Get a reference to the player's inventory
        my_inv = state.inventories[me]

        player_food += my_inv.foodCount

        food_coords = []

        foods = getConstrList(state, None, (FOOD,))

        # Gets a list of all of the food coords
        for food in foods:
            if food.coords[1] < 5:
                food_coords.append(food.coords)

        # coordinates of this AI's tunnel
        tunnel = my_inv.getTunnels()
        t_coords = tunnel[0].coords

        # coordinates of this AI's anthill
        ah_coords = my_inv.getAnthill().coords

        # iterates through ants and scores positioning
        for ant in my_inv.ants:

            # scores queen
            if ant.type == QUEEN:
                queen_health = ant.health

            # scores worker to incentivize food gathering
            elif ant.type == WORKER:

                # tallies up workers
                worker_count += 1

                # if carrying, the closer to the anthill or tunnel, the better
                if ant.carrying:

                    # distance to anthill
                    ah_dist = approxDist(ant.coords, ah_coords)

                    # distance to tunnel
                    t_dist = approxDist(ant.coords, t_coords)

                    # finds closest and scores
                    # if ant.coords == ah_coords or ant.coords == t_coords:
                    # print("PHill")
                    # eval += 100000000
                    if t_dist < ah_dist:
                        w_distance_to_move += t_dist
                    else:
                        w_distance_to_move += ah_dist

                # if not carrying, the closer to food, the better
                else:

                    # distance to foods
                    f1_dist = approxDist(ant.coords, food_coords[0])
                    f2_dist = approxDist(ant.coords, food_coords[1])

                    # finds closest and scores
                    # if ant.coords == food_coords[0] or ant.coords == food_coords[1]:
                    # print("PFood")
                    # eval += 500

                    if f1_dist < f2_dist:
                        w_distance_to_move += f1_dist
                    else:
                        w_distance_to_move += f2_dist

                        # the father from enemy ants, the better
                        # eval += -5 + self.get_closest_enemy_dist(ant.coords, enemy_inv.ants)

            # scores soldiers to incentivize the disruption of the enemy economy
            else:
                # tallies up soldiers
                fighter_count += 1

        return [queen_health, worker_count, w_distance_to_move, fighter_count, player_food]











    ##
    # append_state_to_list
    #
    # Parameters
    #   state - Simplified version of state
    #   utility - Value of state
    #
    # Returns nothing
    #
    ##
    def append_state_to_list(self, state, utility):
        self.state_value_list.append([state, utility])
        return


    ##
    # write_state_list_to_file
    #
    # Parameters
    #   filename - file to open
    #
    # Return
    #   returns nothing
    ##
    def write_state_list_to_file(self, filename):
        with open(filename, 'w') as f:
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
        with open(filename, 'r') as f:
            states = f.read().splitlines()
            for s in states:
                nums = s.split(",")
                for num in nums:
                    num = float(num)
                length = len(s)
                state = s[:length - 1]
                eval = s[length - 1]
                self.state_value_list.append([state, eval])




    ##
    # find_state_eval
    #
    # Parameters:
    #   simplified_state - Consolidated state created from other method
    ##
    def find_state_eval(self, simplified_state):
        for state in self.state_value_list:
            if state[0] == simplified_state:
                return state[1]
        return -25655

    ##
    # append_state
    #
    # Parameters:
    #   simplified_state - Consolidated state created from other method
    ##
    def append_state(self, simplified_state):
        if (self.find_state_eval(simplified_state) == -25655):
            self.state_value_list.append([simplified_state, 0.5])
        else:
            return -25655