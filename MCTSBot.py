import numpy
import Configuration
from math import sqrt, ln
from Bot import Bot
from collections import deque
from numpy import ones, copy

SELECT_CONSTANT = 10 #value from the paper

class Node:
    def __init__(self, parent, value):
        self.score = 0
        self.number_visit = 0
        self.children = []
        self.parent = parent


        if parent is None:
            self.depth = 0
            self.value = [value]
        else:
            self.value = self.parent.value.extend([value])
            self.depth = parent.depth + 1

    def add_child(self, value):
        self.children.append(Node(self, value))

    def back_propagate(self, value):
        '''
        Back propagate the result of the simulation in the branch to the root node
        :param value: 1 if it was a win, 0 for a loss
        '''
        self.score += value
        self.number_visit += 1

        if self.parent is not None:
            self.parent.back_propagate(value)

    def selection(self):
        '''
        selection of the node that will be extended at this iteration
        :return: Nodes
        '''
        node = self.select_node()
        if len(node.children) > 0:
            node = self.selection()
        else:
            return node

    def select_node(self):
        '''
        selection of the next visited nodes for the next extension
        :return: Node
        '''

        max_selection_score = 0
        selected_node = None
        for node in self.children:
            selection_score = node.score + SELECT_CONSTANT * sqrt(ln(self.number_visit) / self.node.number_visit)

            if selection_score > max_selection_score:
                max_selection_score = selection_score
                selected_node = node

        return selected_node

    def initialize_play_out(self, list_players, my_index):
        '''
        Initialize the area with the previous positions of each cycles
        '''

        area = ones(Configuration.MAX_X_GRID + 1, Configuration.MAX_Y_GRID + 1)
        for nb_turn, my_position in enumerate(self.value):
            for player in list_players:
                if player == my_index:
                    area[my_position] = False
                else:
                    position = self.process_random_position(area, previous_position, current_position)

               #TODO: finish the code of the play out initialization

        return area

class MCTSBot():
    def __init__(self):
        self.cur_cycles = {}
        self.wall_cycles = {}
        self.area = ones((30, 20), dtype=bool)
        self.list_players = []
        self.list_players_without_me = []
        self.turn = 0

    def compute_mcts(area, list_players, my_index):
        return ''

    def compute_direction(self, input):
        splitted = input.split('\n')
        nb_players, my_index = [int(i) for i in splitted[0].split()]

        if self.turn == 0:
            for i in range(nb_players):
                self.list_players.append(i)
                self.list_players_without_me.append(i)

            self.list_players_without_me.remove(my_index)

        for i in range(nb_players):
            # x0: starting X coordinate of lightcycle (or -1)
            # y0: starting Y coordinate of lightcycle (or -1)
            # x1: starting X coordinate of lightcycle (can be the same as X0 if you play before this player)
            # y1: starting Y coordinate of lightcycle (can be the same as Y0 if you play before this player)
            x0, y0, x1, y1 = [int(j) for j in splitted[i + 1].split()]

            if x0 != -1:
                self.cur_cycles[i] = (x1, y1)

                if self.turn == 0:
                    self.wall_cycles[i] = [(x0, y0)]
                    self.area[x0, y0] = False

                self.wall_cycles[i].append((x1, y1))
                self.area[x1, y1] = False
            else:
                # If player has lost, remove his wall from the game
                if i in self.cur_cycles:
                    for case in self.wall_cycles[i]:
                        self.area[case] = True

                    del self.cur_cycles[i]
                    del self.wall_cycles[i]

                    self.list_players.remove(i)
                    self.list_players_without_me.remove(i)

        direction = compute_mcts(self.area, self.list_players, my_index)

        return direction