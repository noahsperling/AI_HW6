import random
import sys
import numpy as np


sys.path.append("..")  # so other modules can be found in parent dir
from Player import *
from Constants import *
from Construction import CONSTR_STATS
from Ant import UNIT_STATS
from Move import Move
from GameState import *
from AIPlayerUtils import *


##
# AIPlayer
# Description: This AI player uses Minimax with Alpha/Beta pruning to search for the best move
# based on the evaluate_state heuristic function
# Assignment: Homework #3: Minimax AI
#
# Due Date: April 10th, 2017
#
# @names: Noah Sperling, Avaya Bhattara
##
class AIPlayer(Player):

    #maximum depth
    max_depth = 2

    #current index - for recursive function
    cur_array_index = 0

    #highest evaluated move - to be reset every time the generate_states method is called
    highest_evaluated_move = None

    #highest move score - useful for finding highest evaluated move - to be reset
    highest_move_eval = -1

    #this AI's playerID
    me = -1

    #whether or not the playerID has been set up yet
    me_set_up = False

    # the matrix of weights from the input to the first layer, random starting values - will get overwritten
    first_weight_matrix = np.matrix([[1.0, -2.1, 1.3, -0.5, 2.138, -2.138, 1.2, -0.9, 1.26, 0.11, 2.1, -1.5],
                                     [-0.9, -0.75, 1.75, 0.256, 2.0, -1.25, -1.24, -2.0, 1.98, 0.5, 0.6, 0.7],
                                     [0.1, -0.89, 1.456, 2.013, 0.564, -2.136, 1.789, -0.2, -0.1, 1.03, 0.745, 1.111],
                                     [-2.0, -2.1, 1.023, 0.654, 0.213, -0.5, 0.7, 0.645, 1.236, 2.1, -1.111, 1.756],
                                     [-1.54, 2.113, 0.215, -0.555, -0.721, -1.231, 1.45, -0.2, 1.98, -0.89, -0.7, 1.0],
                                     [0.1, -0.1, 0.23, -0.23, 1.69, 1.420, 2.003, 0.001, 0.592, -0.25, 0.75, 1.0],
                                     [-0.125, 1.1, -0.8, 0.9, 2.131, -2.138, 0.141, 0.56, 1.7, 0.9, 2.0, -2.0],
                                     [1.0, -2.1, 1.45, -0.2, 1.0, 1.1, -0.5, -0.6, -0.7, -0.8, 2.0, 1.1],
                                     [1.1, 0.7, -0.2, 0.9, 1.9, -0.987, -1.2, 2.1, -1.75, 0.4, -1.352, 0.312]])

    # the matrix of weights from the first layer to the output layer, again random starting values
    second_weight_matrix = np.matrix([[3.0984, 1.21, -2.15, 0.6, 1.212, -2.56, -2.0, 1.105, -3.0984]])

    # the learning rate
    alpha = 8.0

    # variables for storing states for training the neural network
    saved_states = []
    state_evals = []


    # __init__
    # Description: Creates a new Player
    #
    # Parameters:
    #   inputPlayerId - The id to give the new player (int)
    ##
    def __init__(self, inputPlayerId):
        super(AIPlayer, self).__init__(inputPlayerId, "Neural Network AI")

        # generates random matrices if uncommented
        # random_array = np.random.rand(9, 12)
        # random_array_2 = np.random.rand(9, 1)
        # self.first_weight_matrix = np.matrix(random_array)
        # for x in range(9):
        #    for y in range(12):
        #        z = random.randint(0, 2)
        #        if z == 0:
        #            self.first_weight_matrix[x, y] = self.first_weight_matrix[x, y] * -1.5 * random.uniform(0, 2)
        #        else:
        #            self.first_weight_matrix[x, y] = self.first_weight_matrix[x, y] * 1.5 * random.uniform(0, 2)
        # for x in range(9):
        #    z = random.randint(0, 2)
        #    if z == 0:
        #        self.first_weight_matrix[x, y] = self.first_weight_matrix[x, 0] * -1.5 * random.uniform(0, 2)
        #    else:
        #        self.first_weight_matrix[x, y] = self.first_weight_matrix[x, 0] * 1.5 * random.uniform(0, 2)
        # print self.first_weight_matrix
        # print self.second_weight_matrix

        # sets matrices to learned weights
        self.first_weight_matrix = np.matrix([[1.44421011e+00,  -1.83471661e-01,   8.76387763e-01,   1.03611718e+01,
                                               -2.58004099e-01,  -3.41595794e-01,   1.88884207e+00,   8.63799627e-01,
                                               1.56357590e-01,   7.71383658e-01,   1.13694462e+00,   2.30197841e+00],
                                              [5.92010098e-01,  -1.61841954e+00,   6.68432695e-01,   2.25298574e+00,
                                               -8.69557890e-02,  -4.36541068e-01,   1.25152057e+00,   9.37014483e-01,
                                               -6.06798411e-01,   4.74948735e-01,   1.18682186e+00,   2.77894235e+00],
                                              [1.35901475e+00,   3.36835709e-01,   4.72625364e-01,   1.45712462e+00,
                                               8.64126536e-03,   8.59257427e-01,   6.40255917e-01,   4.28469006e-01,
                                               8.06547048e-01,   8.78721526e-02,   4.92300046e-01,   2.94331041e-01],
                                              [1.64453615e+00,   8.95026004e-01,   6.18492284e-01,  -3.40111788e-01,
                                               -1.06758656e+00,  -1.42711940e+00,  -1.42783684e-01,   8.14051677e-01,
                                               -3.33957892e-01,   1.42058641e-03,  -1.66977489e+00,  -2.69185274e-01],
                                              [6.60240868e-01,  -4.80800827e-02,   5.74261891e-01,   9.90354839e-01,
                                               -3.93922637e-01,   4.51098401e-02,   2.00529389e+01,   4.95343539e-01,
                                               1.40559193e-02,   2.98765573e-01,  -2.01092459e+01,  -1.09738759e-01],
                                              [-1.43188649e+00,   1.44799239e-01,   6.95284442e-01,  -8.30373730e+00,
                                               2.10347052e-01,   3.44553535e-01,  -1.15463542e+00,   3.00739804e-01,
                                               1.98832831e-01,   6.16184926e-01,  -2.49534078e+00,  -6.80543524e+00],
                                              [1.16037913e+00,   2.00099603e-01,   4.89137990e-01,   1.60226107e+00,
                                               3.18518352e-01,   8.15244087e-02,   4.18534036e-01,   3.74904862e-02,
                                               8.60187790e-01,   8.11351268e-01,   4.88194224e-01,   2.70565428e-01],
                                              [7.86081596e-01,   1.65233407e-01,   6.79081393e-01,  -4.58948257e-02,
                                               -5.82717006e-01,  -9.37887380e-02,  -1.85875057e+01,   6.95652711e-01,
                                               6.42780226e-03,   8.68895417e-01,   1.92201682e+01,  -1.17300153e-01],
                                              [1.39502913e+00,   3.24511058e-01,   9.88146637e-01,   1.56104631e+00,
                                               6.28812310e-02,   6.73812839e-01,   1.01968452e+00,   1.97750506e-01,
                                               6.53715341e-01,   8.89692303e-01,   4.71905993e-01,   3.06223822e-01]])
        self.second_weight_matrix = np.matrix([[3.27069721, 1.38229721, -1.97770279, 0.77229721, 1.38429721,
                                                -2.38770279, -1.82770279, 1.27729721, -2.92610279]])


    ##
    # getPlacement
    #
    # Description: called during setup phase for each Construction that
    #   must be placed by the player.  These items are: 1 Anthill on
    #   the player's side; 1 tunnel on player's side; 9 grass on the
    #   player's side; and 2 food on the enemy's side.
    #
    # Parameters:
    #   construction - the Construction to be placed.
    #   currentState - the state of the game at this point in time.
    #
    # Return: The coordinates of where the construction is to be placed
    ##
    def getPlacement(self, currentState):
        numToPlace = 0
        # implemented by students to return their next move
        if currentState.phase == SETUP_PHASE_1:  # stuff on my side
            numToPlace = 11
            moves = []
            for i in range(0, numToPlace):
                move = None
                while move == None:
                    # Choose any x location
                    x = random.randint(0, 9)
                    # Choose any y location on your side of the board
                    y = random.randint(0, 3)
                    # Set the move if this space is empty
                    if currentState.board[x][y].constr == None and (x, y) not in moves:
                        move = (x, y)
                        # Just need to make the space non-empty. So I threw whatever I felt like in there.
                        currentState.board[x][y].constr == True
                moves.append(move)
            return moves
        elif currentState.phase == SETUP_PHASE_2:  # stuff on foe's side
            numToPlace = 2
            moves = []
            for i in range(0, numToPlace):
                move = None
                while move == None:
                    # Choose any x location
                    x = random.randint(0, 9)
                    # Choose any y location on enemy side of the board
                    y = random.randint(6, 9)
                    # Set the move if this space is empty
                    if currentState.board[x][y].constr == None and (x, y) not in moves:
                        move = (x, y)
                        # Just need to make the space non-empty. So I threw whatever I felt like in there.
                        currentState.board[x][y].constr == True
                moves.append(move)
            return moves
        else:
            return [(0, 0)]


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

        #searches for best move
        selectedMove = self.move_search(currentState, 0, -(float)("inf"), (float)("inf"))

        #if not None, return move, if None, end turn
        if not selectedMove == None:
            return selectedMove
        else:
            return Move(END, None, None)


    ##
    # getAttack
    # Description: Gets the attack to be made from the Player
    #
    # Parameters:
    #   currentState - A clone of the current state (GameState)
    #   attackingAnt - The ant currently making the attack (Ant)
    #   enemyLocation - The Locations of the Enemies that can be attacked (Location[])
    ##
    def getAttack(self, currentState, attackingAnt, enemyLocations):
        # Attack a random enemy.
        return enemyLocations[0]


    ##
    # move_search - recursive
    #
    # uses Minimax with alpha beta pruning to search for best next move
    #
    # Parameters:
    #   game_state - current state
    #   curr_depth - current search depth
    #   alpha      - the parent node's alpha value
    #   beta       - the previous node's beta value
    #
    # Return
    #   returns a move object
    ##
    def move_search(self, game_state, curr_depth, alpha, beta):

        #if max depth surpassed, return state evaluation
        if curr_depth == self.max_depth + 1:
            return self.evaluate_state(game_state)

        #list all legal moves
        move_list = listAllLegalMoves(game_state)

        #remove end turn move if the list isn't empty
        if not len(move_list) == 1:
            move_list.pop()

        # list of nodes, which contain the state, move, and eval
        node_list = []

        #generate states based on moves, evaluate them and put them into a list in node_list
        for move in move_list:
            state_eval = 0
            state = getNextStateAdversarial(game_state, move)
            #state_eval = self.evaluate_state(state)
            #self.saved_states.append(self.generate_input_matrix(state, state_eval, 0))
            #self.state_evals.append(state_eval)
            state_eval = self.generate_input_matrix(state, -1, 1)
            # print(state_eval)
            if not state_eval == 0.00001:
                node_list.append([state, move, state_eval])

        # sorts list of nodes
        self.mergeSort(node_list)

        if not self.me == game_state.whoseTurn:
            move_list.reverse()

        best_nodes = []

        for i in range(0, 2): # temporary
            if not len(node_list) == 0:
                best_nodes.append(node_list.pop())

        #best_val = -1

        #if not at the max depth, expand all the nodes in node_list and return
        if curr_depth <= self.max_depth:
            for node in best_nodes:
                score = self.move_search(node[0], curr_depth + 1, alpha, beta)
                if game_state.whoseTurn == self.me:
                    if score > alpha:
                        alpha = score
                    if alpha >= beta:
                        #print("Pruned")
                        break
                else:
                    if score < beta:
                        beta = score
                    if alpha >= beta:
                        #print("Pruned")
                        break

        #if not curr_depth == 0:
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

            #print(len(best_node))
            if not best_node == []:
                return best_node[1]
            else:
                return None


    ##
    # get_closest_enemy_dist - helper function
    #
    # returns distance to closest enemy from an ant
    #
    # Parameters
    #   my_ant_coords - coordinates of the ant we're comapring enemy locations to
    #   enemy_ants - a list of the enemy ants
    #
    # Return
    #   returns the distance to the closest enemy from my_ant_coords
    ##
    def get_closest_enemy_dist(self, my_ant_coords, enemy_ants):

        # starting value
        closest_dist = 100

        # loops through enemy ants
        for ant in enemy_ants:
            if not ant.type == WORKER:

                # if the distance is the smallest yet, update closest
                dist = approxDist(my_ant_coords, ant.coords)
                if dist < closest_dist:
                    closest_dist = dist

        return closest_dist


    ##
    # get_closest_enemy_worker_dist - helper function
    #
    # returns distance to closest enemy worker ant
    #
    # Parameters
    #   my_ant_coords - coordinates of the ant we're comapring enemy locations to
    #   enemy_ants - a list of the enemy ants
    #
    # Return
    #   returns the distance to the closest enemy worker from my_ant_coords
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
    #
    # Parameters
    #   my_ant_coords - the coordinates of the ant we're comapring the food location to
    #   enemy_food_coords - a list of the coordinates of the enemy food
    #
    ##
    def get_closest_enemy_food_dist(self, my_ant_coords, enemy_food_coords):

        # finds the distance to both enemy foods from the current ant
        enemy_food1_dist = approxDist(my_ant_coords, enemy_food_coords[0])
        enemy_food2_dist = approxDist(my_ant_coords, enemy_food_coords[1])

        # returns the shortest distance
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

        #the starting value, not winning or losing
        eval = 0.0

        #important number
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

        #coordinates of this AI's tunnel
        tunnel = my_inv.getTunnels()
        t_coords = tunnel[0].coords

        #coordinates of this AI's anthill
        ah_coords = my_inv.getAnthill().coords

        #A list that stores the evaluations of each worker
        wEval = []

        #A list that stores the evaluations of each drone, if they exist
        dEval = []

        #queen evaluation
        qEval = 0

        #iterates through ants and scores positioning
        for ant in my_inv.ants:

            #scores queen
            if ant.type == QUEEN:

                qEval = 50.0

                #if queen is on anthill, tunnel, or food it's bad
                if ant.coords == ah_coords or ant.coords == t_coords or ant.coords == food_coords[0] or ant.coords == food_coords[1]:
                    qEval -= 10

                #if queen is out of rows 0 or 1 it's bad
                if ant.coords[0] > 1:
                    qEval -= 10

                # the father from enemy ants, the better
                qEval -= 2 * self.get_closest_enemy_dist(ant.coords, enemy_inv.ants)

            #scores worker to incentivize food gathering
            elif ant.type == WORKER:

                #tallies up workers
                worker_count += 1

                #if carrying, the closer to the anthill or tunnel, the better
                if ant.carrying:

                    wEval.append(100.0)

                    #distance to anthill
                    ah_dist = approxDist(ant.coords, ah_coords)

                    #distance to tunnel
                    t_dist = approxDist(ant.coords, t_coords)

                    #finds closest and scores
                    #if ant.coords == ah_coords or ant.coords == t_coords:
                        #print("PHill")
                        #eval += 100000000
                    if t_dist < ah_dist:
                        wEval[worker_count - 1] -= 5 * t_dist
                    else:
                        wEval[worker_count - 1] -= 5 * ah_dist

                #if not carrying, the closer to food, the better
                else:

                    wEval.append(80.0)

                    #distance to foods
                    f1_dist = approxDist(ant.coords, food_coords[0])
                    f2_dist = approxDist(ant.coords, food_coords[1])

                    #finds closest and scores
                    #if ant.coords == food_coords[0] or ant.coords == food_coords[1]:
                        #print("PFood")
                        #eval += 500

                    if f1_dist < f2_dist:
                        wEval[worker_count - 1] -= 5 * f1_dist
                    else:
                        wEval[worker_count - 1] -= 5 * f2_dist

                #the father from enemy ants, the better
                #eval += -5 + self.get_closest_enemy_dist(ant.coords, enemy_inv.ants)

            #scores soldiers to incentivize the disruption of the enemy economy
            else:

                #tallies up soldiers
                drone_count += 1

                dEval.append(50.0)

                nearest_enemy_worker_dist = self.get_closest_enemy_worker_dist(ant.coords, enemy_inv.ants)

                #if there is an enemy worker
                if not nearest_enemy_worker_dist == 100:
                     dEval[drone_count - 1] -= 5 * nearest_enemy_worker_dist

                #if there isn't an enemy worker, go to the food
                else:
                    dEval[drone_count - 1] -= 5 * self.get_closest_enemy_food_dist(ant.coords, enemy_food_coords)

        #scores other important things

        #state eval
        sEval = 0

        #assesses worker inventory
        if worker_count == 2:
            sEval += 50
        elif worker_count < 2:
            sEval -= 10
        elif worker_count > 2:
            eval_num = 0.00001
            return eval_num

        #assesses drone inventory
        if drone_count == 2:
            sEval += 50
        elif drone_count < 2:
            sEval -= 10
        elif drone_count > 2:
            eval_num = 0.00001
            return eval_num

        #assesses food
        if not sEval == 0:
            sEval += 20 * my_inv.foodCount

        #a temporary variable
        temp = 0

        #scores workers
        for val in wEval:
            temp += val
        if worker_count == 0:
            wEvalAv = 0
        else:
            wEvalAv = temp/worker_count

        temp = 0

        #scores drones
        for val in dEval:
            temp += val

        if not drone_count == 0:
            dEvalAv = temp/drone_count
        else:
            dEvalAv = 0

        #total possible score
        total_possible = 100.0 + 50.0 + 50.0 + 300.0

        #scores total evaluation and returns
        eval = (qEval + wEvalAv + dEvalAv + sEval)/ total_possible
        if eval <= 0:
            eval = 0.00002

        # print "wste called from heuristic"
        # self.write_state_to_file(state, eval, "states_laptop.txt")
        # self.generate_input_matrix(state, eval)

        return eval


    ##
    # generate_input_matrix
    #
    # generates an input matrix for the neural network
    #
    # Parameters
    #   state - the gamestate object to assess
    #   h_eval - the heuristic function's evaluation of the state
    #          - to be ignored after training
    #
    # Return:
    #   returns a double between zero and one
    #
    ##
    def generate_input_matrix(self, state, h_eval, matrix_or_eval):

        # food inventory numbers
        player_food = 0
        enemy_food = 0

        # queen info
        queen_health = 0
        queen_dist_to_closest_enemy = 0
        queen_on_construct_not_grass = 0
        queen_in_rows_0_or_1 = 0

        # worker info
        worker_count = 0
        num_carrying = 0
        w_distance_to_move = 0
        distance_to_food = 0

        # drone info
        drone_count = 0
        d_distance_to_move = 0

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

        player_food += my_inv.foodCount
        enemy_food += enemy_inv.foodCount

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

        # iterates through ants and scores positioning
        for ant in my_inv.ants:

            # scores queen
            if ant.type == QUEEN:

                # if queen is on anthill, tunnel, or food it's bad
                if ant.coords == ah_coords or ant.coords == t_coords or ant.coords == food_coords[0] or ant.coords == \
                        food_coords[1]:
                    queen_on_construct_not_grass += 1

                # if queen is out of rows 0 or 1 it's bad
                if ant.coords[0] > 1:
                    queen_in_rows_0_or_1 += 1

                # the father from enemy ants, the better
                queen_dist_to_closest_enemy += self.get_closest_enemy_dist(ant.coords, enemy_inv.ants)

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
                drone_count += 1

                nearest_enemy_worker_dist = self.get_closest_enemy_worker_dist(ant.coords, enemy_inv.ants)

                # if there is an enemy worker
                if not nearest_enemy_worker_dist == 100:
                    d_distance_to_move += nearest_enemy_worker_dist

                # if there isn't an enemy worker, go to the food
                else:
                    d_distance_to_move += self.get_closest_enemy_food_dist(ant.coords, enemy_food_coords)

        #print(player_food, enemy_food, queen_health, queen_dist_to_closest_enemy, queen_on_construct_not_grass,
        #      queen_in_rows_0_or_1, worker_count, num_carrying, w_distance_to_move, distance_to_food, drone_count,
        #      d_distance_to_move)
        output = np.matrix([[player_food], [enemy_food], [queen_health], [queen_dist_to_closest_enemy],
                [queen_on_construct_not_grass], [queen_in_rows_0_or_1], [worker_count], [num_carrying],
                [w_distance_to_move], [distance_to_food], [drone_count], [d_distance_to_move]])
        if matrix_or_eval == 0:
            return output
        else:
            return self.neural_network(output, h_eval)


    ##
    # neural_network
    #
    # simulates the neural network based on the input matrix and learns the
    # heuristic function
    #
    # Parameters
    #   input_matrix - a matrix generated by the self
    #
    # Return
    #   returns a state evaluation from the neural network
    #
    ##
    def neural_network(self, input_matrix, h_eval):

        # calculates input for each node in the first laer of the neural network
        first_layer_input = np.matmul(self.first_weight_matrix, input_matrix)

        temp_list = []

        # calculates ouput for first layer perceptrons
        for x in np.nditer(first_layer_input):
            temp_list.append(self.g(x))

        # converts outputs to a matrix
        first_layer_output = np.matrix([[temp_list[0]], [temp_list[1]], [temp_list[2]], [temp_list[3]], [temp_list[4]],
                                        [temp_list[5]], [temp_list[6]], [temp_list[7]], [temp_list[8]]])

        #print self.second_weight_matrix
        #print first_layer_output

        # calculates input to second layer
        second_layer_input = np.matmul(self.second_weight_matrix, first_layer_output)

        #calculates single output
        second_layer_output = self.g(second_layer_input[0, 0])

        # this stores output in an inefficient way that works so I'm not going to worry about it
        output = 0

        for y in np.nditer(second_layer_output):
            output += y
            break

        # print output

        # calculates error
        error = h_eval - output

        #print "Error:", error

        #if not intended to learn, return output
        if h_eval == -1:
            return output

        # back propogation if training

        index = 0

        delta_array = []

        swl = []

        # back propogation for second weight matrix
        for x in np.nditer(self.second_weight_matrix):
            delta = self.g_derivative(second_layer_input, False) * error
            #g_deriv = self.g_derivative(second_layer_input, True)
            new_weight = x + self.alpha * delta * output
            swl.append(new_weight)
            index += 1

        # re-forms second weight matrix
        self.second_weight_matrix = np.matrix([[swl[0], swl[1], swl[2], swl[3], swl[4], swl[5], swl[6], swl[7],
                                                swl[8]]])

        # a list of the new weights
        weight_list = []

        # stores the list of new weights between first and second layer of perceptrons
        for item in np.nditer(self.second_weight_matrix):
            weight_list.append(item)

        # makes an array of delta values
        for x in range(9):
            delta_array.append(self.g_derivative(first_layer_output[x], True) * weight_list[x] * delta)

        # back propogation for the first weight matrix
        for x in range(9):
            for y in range(12):
                self.first_weight_matrix[x, y] = self.first_weight_matrix[x, y] + self.alpha * delta_array[x] * input_matrix[y, 0]

        return output


    ##
    # g(x)
    #
    # Parameters
    #   x - number to input to sigmoid function
    #
    # Return
    #   returns the output value of the sigmoid function
    ##
    def g(self, x):
        return 1 / (1 + 2.7182818284 ** (-1 * x))


    ##
    # g_derivative
    #
    # Parameters
    #   x - number to input to sigmoid function
    #
    # Return
    #   returns the derivative of the sigmoid function at x
    ##
    def g_derivative(self, x, g_of_x):

        # calculates differently based on whether g(x) was inputted or just x
        if not g_of_x:
            output = (1 / (1+np.exp(-x).item())) * (1 - (1 / (1+np.exp(-x).item())))
            return output
        else:
            return x * (1 - x)


    ##
    # merge_sort
    #
    # useful for sorting the move list from least to greatest in nlog(n) time
    #
    # Parameters:
    #   alist - a list to sort
    #
    # Return
    #   returns a sorted list
    ##
    def mergeSort(self, alist):
        if len(alist) > 1:

            # splits list
            mid = len(alist) // 2
            lefthalf = alist[:mid]
            righthalf = alist[mid:]

            # recursive call
            self.mergeSort(lefthalf)
            self.mergeSort(righthalf)

            # values to keep track of indices
            i = 0
            j = 0
            k = 0

            # sorts lists
            while i < len(lefthalf) and j < len(righthalf):
                if lefthalf[i][2] < righthalf[j][2]:
                    alist[k] = lefthalf[i]
                    i += 1
                else:
                    alist[k] = righthalf[j]
                    j += 1
                k += 1

            while i < len(lefthalf):
                alist[k] = lefthalf[i]
                i += 1
                k += 1

            while j < len(righthalf):
                alist[k] = righthalf[j]
                j += 1
                k += 1


    ##
    # write_state_to_file
    #   writes a gamestate to a file in a format easily transferred to an input matrix
    #   for the neural network
    #
    # Parameters
    #   state - the gamestate to save
    #   state_eval - the evaluation of the state by the heuristic
    #   file_name - the file path to write the states to
    #
    # Returns nothing
    ##
    def write_state_to_file(self, state, state_eval, file_name):

        # the input matrix to save
        values = self.generate_input_matrix(state, state_eval, 0)

        # a list of values to print the matrix more easily
        lv = []

        # loops through the matrix and adds the values to a list
        for v in np.nditer(values):
            lv.append(v)

        # checks to make sure nothing went wrong generating an input matrix
        if not len(lv) == 12:
            print "Error"

        # if nothing went wrong, writes the state to the specified file
        else:
            try:
                file = open(file_name, "a")
                # print "File opened"
                file.write("%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n" % (lv[0], lv[1], lv[2], lv[3], lv[4], lv[5], lv[6], lv[7], lv[8], lv[9], lv[10], lv[11], state_eval))
                file.close()
                # print "File closed"

            # just in case
            except Exception:
                print("Something went wrong opening and appending to the text file.")
        return


    ##
    # read_states_from_file_and_train_neural_network
    #   does what the name indicates
    #
    # Parameters
    #   file_name - file path to read from and train network
    #
    # Returns nothing
    ##
    def read_states_from_file_and_train_neural_network(self, file_name):

        # counts number of states to train
        states_to_train = sum(1 for line in open(file_name))

        # opens file to read from
        with open(file_name) as f:

            # prints weight matrix before training for comparison
            print self.first_weight_matrix

            # to keep track of how many states assessed
            line_count = 0

            # loops through lines in the file and trains the neural network
            for line in f:

                # splits comma separated values into a list
                ls = line.split(",")

                # a second list for float values
                ls2 = []

                # loops through ls, converts strings to floats, and appends to ls2
                for num in ls:
                    ls2.append(float(num))

                # creates input matrix for a state and finds the state's evaluation
                input_matrix = np.matrix([[ls2[0]], [ls2[1]], [ls2[2]], [ls2[3]], [ls2[4]], [ls2[5]], [ls2[6]],
                                          [ls2[7]], [ls2[8]], [ls2[9]], [ls2[10]], [ls2[11]]])
                state_eval = ls2[12]

                # calls the neural network so that it learns
                self.neural_network(input_matrix, state_eval)

                # iterate line count
                line_count += 1

                # decrease alpha over time
                if line_count == 1000:
                    self.alpha = 2
                elif line_count == 10000:
                    self.alpha = 1
                elif self.alpha == 20000:
                    self.alpha = 0.5
                elif self.alpha == 50000:
                    self.alpha = 0.1

                # prints weight matrices at end of learning
                if line_count >= states_to_train - 1:
                    print self.first_weight_matrix
                    print self.second_weight_matrix
                    break

        # closes file and returns
        f.close()
        return

    ##
    # shuffle_states_in_file
    #   reads lines from a file and shuffles their order, outputting to a new file
    #
    # Parameters
    #   file_name - file path to read from
    #
    # Returns nothing
    ##
    def shuffle_states_in_file(self, file_name):

        # a list for the gamestates saved in the file
        state_list = []

        # opens file and appends states to the states list
        with open(file_name) as f:
            for line in f:
                state_list.append(line)

        # shuffles the order of state_list
        random.shuffle(state_list)

        #writes the new states to a new file
        with open("C:/Users/theem/PycharmProjects/AI_HW5/states_shuffled_2.txt", "w") as wf:
            for state in state_list:
                wf.write(state)


    ##
    # registerWin
    #   overrides registerWin method to train neural network after a game (when not commented out)
    #
    # Parameters
    #   hasWon - a boolean saying if this AI has won
    #
    # Returns nothing
    ##
    #def registerWin(self, hasWon):
    #    super(AIPlayer, self).registerWin(hasWon)
    #
    #    f = open("C:/Users/theem/PycharmProjects/AI_HW5/states_2.txt", "a")
    #
    #    index = 0
    #
    #    for state in self.saved_states:
    #        f.write("%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n" % (state[0, 0], state[1, 0], state[2, 0], state[3, 0], state[4, 0], state[5, 0], state[6, 0],
    #                                                              state[7, 0], state[8, 0], state[9, 0], state[10, 0], state[11, 0], self.state_evals[index]))
    #        index += 1
    #    f.close()
    #
    #
    #    self.shuffle_states_in_file("C:/Users/theem/PycharmProjects/AI_HW5/states_2.txt")
    #
    #    self.read_states_from_file_and_train_neural_network("C:/Users/theem/PycharmProjects/AI_HW5/states_shuffled_2.txt")






# AIP = AIPlayer(PLAYER_ONE)
# AIP.shuffle_states_in_file("C:/Users/theem/PycharmProjects/AI_HW5/states.txt")
# AIP.read_states_from_file_and_train_neural_network("C:/Users/theem/PycharmProjects/AI_HW5/states_shuffled.txt")
# AIP.read_states_from_file_and_train_neural_network("C:/Users/Noah/PycharmProjects/AI_HW5/states_laptop.txt")

# unit tests
# testPlayer = AIPlayer(PLAYER_ONE)
#test get_closest_enemy_dist
# testAntList = [Ant((2,4), 4, None), Ant((3,5), 2, None), Ant((2,5), 3, None), Ant((2,2), 1, None)]
# val = AIPlayer.get_closest_enemy_dist(testPlayer, (2,1), testAntList)
# assert (AIPlayer.get_closest_enemy_dist(testPlayer, (2,1), testAntList)==3), "get_closest_enemy_dist isn't working right(returned %d)" % val

# test get_closest_enemy_worker_dist
# testAntList = [Ant((2,4), 1, None), Ant((3,5), 1, None), Ant((2,5), 1, None), Ant((2,2), 2, None)]
# val = AIPlayer.get_closest_enemy_worker_dist(testPlayer, (2,1), testAntList)
# assert (AIPlayer.get_closest_enemy_worker_dist(testPlayer, (2,1), testAntList)==3), "get_closest_enemy_worker_dist isn't working right(returned %d)" % val

# test get_closest_enemy_food_dist
# val = AIPlayer.get_closest_enemy_food_dist(testPlayer, (2,3), [(2,4), (2,5)])
# assert (AIPlayer.get_closest_enemy_food_dist(testPlayer, (2,3), [(2,4), (2,5)])==1), "get_closest_enemy_food_dist isn't working right(returned %d)" % val

# test evaluate_state
# board = [[Location((col, row)) for row in xrange(0,BOARD_LENGTH)] for col in xrange(0,BOARD_LENGTH)]
# testConstrList1=[Construction((1,1), ANTHILL), Construction((1,2), TUNNEL), Construction((9,1), FOOD), Construction((9,2), FOOD)]
# testConstrList2=[Construction((9,9), ANTHILL), Construction((9,8), TUNNEL), Construction((1,8), FOOD), Construction((1,9), FOOD)]
# p1Inventory = Inventory(PLAYER_ONE, [Ant((1,1), 0, PLAYER_ONE), Ant((1,5), 1, PLAYER_ONE)], testConstrList1, 0)
# p2Inventory = Inventory(PLAYER_TWO, [Ant((1,2), 2, PLAYER_ONE), Ant((1,6), 2, PLAYER_ONE)], testConstrList2, 0)
# neutralInventory = Inventory(NEUTRAL, [], [], 0)
# testState1 = GameState(board, [p1Inventory, p2Inventory, neutralInventory], MENU_PHASE, PLAYER_ONE)
# eval1 = AIPlayer.evaluate_state(testPlayer, testState1)
# board = [[Location((col, row)) for row in xrange(0,BOARD_LENGTH)] for col in xrange(0,BOARD_LENGTH)]
# p1Inventory = Inventory(PLAYER_ONE, [Ant((1,1), 2, PLAYER_ONE), Ant((1,5), 2, PLAYER_ONE)], [Construction((1,1), ANTHILL), Construction((1,2), TUNNEL)], 0)
# p2Inventory = Inventory(PLAYER_TWO, [Ant((1,2), 0, PLAYER_ONE), Ant((1,6), 1, PLAYER_ONE)], [Construction((9,9), ANTHILL), Construction((9,8), TUNNEL)], 0)
# neutralInventory = Inventory(NEUTRAL, [], [], 0)
# testState2 = GameState(board, [p1Inventory, p2Inventory, neutralInventory], MENU_PHASE, PLAYER_ONE)
# eval2 = AIPlayer.evaluate_state(testPlayer, testState2)
# assert(eval1<eval2), "evaluate_state is broken (returned %d and %d)" % (eval1, eval2)


