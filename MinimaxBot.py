import sys
from math import sqrt, log
from random import randint
from time import clock
from Utils import compute_path, detect_articulation_points, compute_tree_of_chambers, compute_voronoi, generate_index_cache, generate_manhattan_cache


class Node:
    def __init__(self, p_parent, p_is_my_turn, p_area, p_previous_position, p_new_position, p_is_separeted):
        self.score = -1
        self.is_my_turn = p_is_my_turn
        self.is_separeted = p_is_separeted

        self.parent = p_parent
        self.children = []
        self.depth = p_parent.depth + 1

        self.area = p_area

        if self.is_my_turn:
            self.my_prev_position = p_previous_position
            self.my_position = p_new_position
        else:
            self.ennemy_prev_position = p_previous_position
            self.ennemy_position = p_new_position

    def add_child(self, previous_positions, new_position, is_separeted):
        child = Node(self, not self.is_my_turn, previous_positions, new_position, is_separeted)
        self.children.append(child)
        return child

    def expansion(self, context_area, index_cache):
        '''
        Create the next turn of the game
        Forbid to add a previous move done by the player in the tree
        and done by all players in previous simulation step
        '''

        area = context_area[:]

        if self.is_my_turn:
            area[index_cache[self.my_position[0]][self.my_position[1]]] = False
            previous_position = self.ennemy_prev_position
        else:
            area[index_cache[self.ennemy_position[0]][self.ennemy_position[1]]] = False
            previous_position = self.my_prev_position

        if self.depth < 5:
            offsets = [[1, 0], [-1, 0], [0, 1], [0, -1]]
            for off_x, off_y in offsets:
                new_x = previous_position[0] + off_x
                new_y = previous_position[1] + off_y

                if 0 <= new_x < 30 and 0 <= new_y < 20 and area[index_cache[new_x][new_y]]:
                    self.add_child(previous_position, (new_x, new_y), self.is_separeted)
        else:
            self.score = self.evaluate()

    def back_propagate(self, value, space):
        '''
        Back propagate the result of the simulation in the branch to the root node
        :param value: 1 if it was a win, 0 for a loss
        '''

        if self.is_my_turn:
            best_score = 0
            for node in self.children:
                if node.score > best_score:
                    best_score = node.score
            return best_score
        else:
            min_score = 1000
            for node in self.children:
                if min_score > node.score:
                    min_score = node.score
            return min_score

        if self.parent is not None:
            self.parent.back_propagate(value, space)

    def selection(self):
        '''
        selection of the node that will be played by the bot
        :return: Node
        '''

        winner = None
        for node in self.children:
            if node.score == self.score:
                winner = node
                break

        return winner

    def evaluate(self):
        '''
        Evaluate the score of the leaf (based on the space remaining for the players)
        '''
        score = 0
        return score


def compute_Minimax(area, cur_cycles, previous_moves, list_players, my_index, manhattan_cache, index_cache):
    '''
    Compute the MCTS tree from the current situation
    :return: direction to take for this simulation step
    '''

    current_position = cur_cycles[my_index]
    tree = Node(None, current_position, False)
    tree.expansion(area,index_cache)

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

        direction = compute_Minimax(self.area, self.current_move, self.wall_cycles, self.list_players, my_index, self.manhattan_cache, self.index_cache)

        print(direction, flush=True)
        self.turn += 1
        return direction