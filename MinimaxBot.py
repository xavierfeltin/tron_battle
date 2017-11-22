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

        self.position = p_new_position
        self.prev_positions = {}
        if p_is_my_turn:
            self.prev_positions[0] = p_previous_positions[0]
            self.prev_positions[1] = p_parent.position
        else:
            self.prev_positions[0] = p_parent.position
            self.prev_positions[1] = p_previous_positions[1]

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
                    new_x = self.position[0] + off_x
                    new_y = self.position[1] + off_y

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

        return winner

    def evaluate(self):
        '''
        Evaluate the score of the leaf (based on the space remaining for the players)
        '''
        score = 0
        return score


def minimax(node, depth, is_maximizing_player):
    '''
    Compute the minimax of the node set in parameter
    maximise or minimize function of the argument
    :return: the best value for the player to play this turn
    '''
    if depth == 0 or node.is_end_game:
        return node.evaluate()

    if is_maximizing_player:
        bestValue = -inf
        for child in node.children:
            score = minimax(child, depth-1, False)
            bestValue = max(bestValue, score)
        return bestValue

    else:
        bestValue = inf
        for child in node.children:
            score = minimax(child, depth-1, True)
            bestValue = min(bestValue, score)
        return bestValue

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


        tree = Node(None, True, self.area, self.previous_move, self.current_move[0], False)
        tree.expand(self.index_cache, 5)
        #minimax(tree, 5, True)

        print(direction, flush=True)
        self.turn += 1
        return direction