import sys
from math import sqrt, log, inf
from random import randint
from time import clock
from Utils import compute_path, detect_articulation_points, compute_tree_of_chambers, compute_voronoi, generate_index_cache, generate_manhattan_cache


class Node:
    def __init__(self, p_parent, p_is_my_turn, p_area, p_previous_positions, p_new_position, p_is_separeted):
        self.score = -1
        self.is_my_turn = p_is_my_turn
        self.is_separeted = p_is_separeted

        self.parent = p_parent
        self.children = []

        self.area = p_area[:]


        self.positions = {}
        self.prev_positions = {}
        if p_parent is None:
            #root case: current state of the game
            self.prev_positions[0] = p_previous_positions[0]
            self.positions[0] = p_new_position[0]

            self.prev_positions[1] = p_previous_positions[1]
            self.positions[1] = p_new_position[1]

        else:
            if p_is_my_turn:
                #Ennemy just played
                self.prev_positions[1] = p_parent.positions[1]
                self.positions[1] = p_new_position

                #Player conserves last played positions
                self.prev_positions[0] = p_parent.prev_positions[0]
                self.positions[0] = p_parent.positions[0]
            else:
                #Player just played
                self.prev_positions[0] = p_parent.positions[0]
                self.positions[0] = p_new_position

                #Ennemy conserves last played positions
                self.prev_positions[1] = p_parent.prev_positions[1]
                self.positions[1] = p_parent.positions[1]

        for prev_position in p_previous_positions:
            self.area[prev_position] = False

    def add_child(self, previous_positions, new_position, is_separeted):
        child = Node(self, not self.is_my_turn, self.area, previous_positions, new_position, is_separeted)
        self.children.append(child)
        return child

    def expand(self, index_cache, p_depth):
        '''
        Create the next turn of the game
        Forbid to add a previous move done by the player in the tree
        and done by all players in previous simulation step
        '''

        depth = p_depth
        front_nodes = [self]
        nb_children = 1

        while(depth > 0 and nb_children > 0):

            nb_children = 0
            next_nodes = []
            for node in front_nodes:
                offsets = [[1, 0], [-1, 0], [0, 1], [0, -1]]
                nb_node_children = 0
                for off_x, off_y in offsets:
                    if node.is_my_turn:
                        #Generate players next moves
                        pos = self.positions[0]
                    else:
                        #Generate ennemy next moves
                        pos = self.positions[1]

                    new_x = pos[0] + off_x
                    new_y = pos[1] + off_y

                    if 0 <= new_x < 30 and 0 <= new_y < 20 and node.area[index_cache[new_x][new_y]]:
                        child = node.add_child(node.prev_positions, (new_x, new_y), node.is_separeted)
                        next_nodes.append(child)
                        nb_node_children += 1

                if nb_node_children > 0:
                    nb_children += nb_node_children
                else:
                    node.is_end_game = True

            if nb_children > 0:
                front_nodes = next_nodes

            depth -= 1

    def selection(self, p_score):
        '''
        selection of the node that will be played by the bot
        :return: Node
        '''

        winner = None
        for child in self.children:
            if child.score == p_score:
                winner = child
                break

        return winner.positions[0]

    def evaluate(self, manhattan_cache, index_cache):
        '''
        Evaluate the score of the leaf (based on the space remaining for the players)
        '''

        my_index = 0 #TODO: make my_index a variable like for real game

        msg = ''
        start = clock()
        voronoi_area, voronoi_spaces = compute_voronoi(self.area, self.positions, [0, 1], index_cache)
        voronoi_time = (clock() - start) * 1000
        msg += 'voronoi: ' + str(round(voronoi_time,2)) + 'ms'

        if not self.is_separeted:
            #r_x, r_y = self.positions[my_index][0], self.positions[my_index][1]
            #g_x, g_y = self.positions[1 - my_index][0], self.positions[1 - my_index][1]

            #start = clock()
            #distance = compute_path(self.area, self.positions[my_index], index_cache[r_x][r_y],
            #                        self.positions[1 - my_index], index_cache[g_x][g_y], manhattan_cache,
            #                        index_cache)
            #path_time = (clock() - start) * 1000
            # msg += ', path: ' + str(round(path_time,2)) + 'ms'

            if voronoi_spaces[2] == 0:
                self.is_separeted = True
                # self.is_end_game = True

        if self.is_separeted:
            start = clock()
            my_articulation_points = detect_articulation_points(self.area, self.positions[my_index],
                                                                index_cache[self.positions[my_index][0]][
                                                                    self.positions[my_index][1]], index_cache)
            ennemy_articulation_points = detect_articulation_points(self.area, self.positions[1 - my_index],
                                                                    index_cache[self.positions[1 - my_index][0]][
                                                                        self.positions[1 - my_index][1]],
                                                                    index_cache)
            articulation_separated_time = (clock() - start) * 1000
            msg += ', AP sep: ' + str(round(articulation_separated_time,2)) + 'ms'
        else:
            start = clock()
            my_articulation_points = detect_articulation_points(self.area, self.positions[my_index],
                                                                index_cache[self.positions[my_index][0]][
                                                                    self.positions[my_index][1]], index_cache)
            ennemy_articulation_points = my_articulation_points
            articulation_time = (clock() - start) * 1000
            msg += ', AP: ' + str(round(articulation_time,2)) + 'ms'

        is_in_territory = False
        if self.is_separeted:
            is_in_territory = len(my_articulation_points) > 0
        else:
            for articulation in my_articulation_points:
                if voronoi_area[articulation] == my_index:
                    is_in_territory = True
                    break

        if is_in_territory:
            start = clock()
            my_spaces = compute_tree_of_chambers(self.area, voronoi_area, my_articulation_points,
                                                 self.positions[my_index], self.prev_positions[0], index_cache, my_index)
            tree_time = (clock() - start) * 1000
            msg += ', My tree: ' + str(round(tree_time,2)) + 'ms'
        else:
            my_spaces = voronoi_spaces[my_index]

        is_in_territory = False
        if self.is_separeted:
            is_in_territory = len(ennemy_articulation_points) > 0
        else:
            for articulation in ennemy_articulation_points:
                if voronoi_area[articulation] == (1 - my_index):
                    is_in_territory = True
                    break

        if is_in_territory:
            start = clock()
            ennemy_spaces = compute_tree_of_chambers(self.area, voronoi_area, ennemy_articulation_points,
                                                     self.positions[1 - my_index], self.prev_positions[1], index_cache,
                                                     (1 - my_index))
            tree_time = (clock() - start) * 1000
            msg += ', Ennemy tree: ' + str(round(tree_time,2)) + 'ms'
        else:
            ennemy_spaces = voronoi_spaces[1 - my_index]

        print(msg, flush=True)

        if my_spaces == 0:
            return -inf
        elif ennemy_spaces == 0:
            return inf
        else:
            self.score = my_spaces-ennemy_spaces


def minimax(node, alpha, beta, depth, is_maximizing_player, manhattan_cache, index_cache):
    '''
    Compute the minimax of the node set in parameter
    maximise or minimize function of the argument
    Add alpha-beta pruning
    :return: the best value for the player to play this turn
    '''

    if depth == 0 or len(node.children) == 0:
        node.evaluate(manhattan_cache, index_cache)
        return node.score

    if is_maximizing_player:
        best_value = -inf
        for child in node.children:
            score = minimax(child, alpha, beta, depth-1, False, manhattan_cache, index_cache)
            best_value = max(best_value, score)

            if best_value >= beta:
                node.score = best_value
                return best_value
            alpha = max(alpha, best_value)
        #node.score = best_value
        #return best_value

    else:
        best_value = inf
        for child in node.children:
            score = minimax(child, alpha, beta, depth-1, True, manhattan_cache, index_cache)
            best_value = min(best_value, score)

            if alpha >= best_value:
                node.score = best_value
                return best_value
            beta = min(beta, best_value)

        #node.score = best_value
        #return best_value

    node.score = best_value
    return best_value

class MinimaxBot():
    def __init__(self):
        self.current_move = {}
        self.wall_cycles = {}
        self.area = [True] * 600
        self.list_players = []
        self.list_players_without_me = []
        self.turn = 0

        self.previous_move = {}

        self.manhattan_cache = generate_manhattan_cache()
        self.index_cache = generate_index_cache()

    def compute_direction(self, input):
        splitted = input.split('\n')
        nb_players, my_index = [int(i) for i in splitted[0].split()]

        if self.turn == 0:
            for i in range(nb_players):
                self.list_players.append(i)
                self.list_players_without_me.append(i)

            self.list_players_without_me.remove(my_index)

        for i in range(nb_players):
            # x0: starting X coordinate of lightcycle (or -1 if lost)
            # y0: starting Y coordinate of lightcycle (or -1 if lost)
            # x1: starting X coordinate of lightcycle (can be the same as X0 if you play before this player on the second turn)
            # y1: starting Y coordinate of lightcycle (can be the same as Y0 if you play before this player on the second turn)
            x0, y0, x1, y1 = [int(j) for j in splitted[i + 1].split()]

            if x0 != -1:
                self.current_move[i] = (x1, y1)

                if self.turn == 0:
                    self.previous_move[i] = (-1,-1)
                    self.current_move[i] = (x0, y0)
                    self.wall_cycles[i] = [(x0, y0)]
                    self.area[self.index_cache[x0][y0]] = False
                else:
                    self.previous_move[i] = self.current_move[i]
                    self.current_move[i] = (x1,y1)
                    self.wall_cycles[i].append((x1, y1))
                    self.area[self.index_cache[x1][y1]] = False
            else:
                # If player has lost, remove his wall from the game
                if i in self.current_move:
                    for case in self.wall_cycles[i]:
                        self.area[self.index_cache[case[0]][case[1]]] = False

                    del self.current_move[i]
                    del self.wall_cycles[i]

                    self.list_players.remove(i)
                    self.list_players_without_me.remove(i)

        r_x = self.current_move[0][0]
        r_y = self.current_move[0][1]
        g_x = self.current_move[1][0]
        g_y = self.current_move[1][1]

        distance = compute_path(self.area, self.current_move[my_index], self.index_cache[r_x][r_y],
                                           self.current_move[1 - my_index], self.index_cache[g_x][g_y], self.manhattan_cache,
                                           self.index_cache)

        MAX_DEPTH = 7
        start = clock()
        tree = Node(None, True, self.area, self.previous_move, self.current_move, distance is None)
        tree.expand(self.index_cache, MAX_DEPTH)
        print('time to expand: ' + str((clock()-start)*1000), flush=True)

        score = minimax(tree, -inf, inf, MAX_DEPTH, True, self.manhattan_cache, self.index_cache)
        next_position = tree.selection(score)

        current_position = self.current_move[0]
        direction = ''
        if next_position[0] - current_position[0] > 0: direction = 'RIGHT'
        elif next_position[0] - current_position[0] < 0: direction = 'LEFT'
        elif next_position[1] - current_position[1] > 0: direction = 'DOWN'
        elif next_position[1] - current_position[1] < 0: direction = 'UP'

        self.turn += 1

        print(direction, flush=True)
        return direction