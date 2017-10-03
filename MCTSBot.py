import numpy
import Configuration

from math import sqrt, log
from random import randint
from Bot import Bot
from collections import deque
from numpy import ones, copy

SELECT_CONSTANT = 10 #value from the paper
NB_TURNS_CHECK = 5 #value from the paper
A = 1 #value from the paper
NB_MCTS_ITERATIONS = 800 #experimental


class Node:
    def __init__(self, parent, value):
        self.score = 0
        self.number_visit = 1
        self.children = []
        self.parent = parent

        if parent is None:
            self.depth = 0
            self.value = [value]
        else:
            self.value = self.parent.value[:]
            self.value.append(value)
            self.depth = parent.depth + 1

    def add_child(self, value):
        self.children.append(Node(self, value))

    def back_propagate(self, value):
        '''
        Back propagate the result of the simulation in the branch to the root node
        :param value: 1 if it was a win, 0 for a loss
        '''
        if value: self.score += 1
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
            node = node.selection()
        else:
            return node

    def select_node(self):
        '''
        selection of the next visited nodes for the next extension
        :return: Node
        '''

        max_selection_score = 0
        selected_node = None

        if len(self.children) > 0:
            for node in self.children:
                selection_score = node.score + SELECT_CONSTANT * sqrt(log(self.number_visit) / node.number_visit)

                if selection_score >= max_selection_score:
                    max_selection_score = selection_score
                    selected_node = node

            return selected_node
        else:
            return self

    def initialize_play_out(self, context_area, current_positions, players, my_index):
        '''
        Initialize the area with the previous positions of each cycles
        :return: the area at the current state of the game
        :return: the last positions of each player
        '''

        area = numpy.copy(context_area)
        walls = {}
        for player in players:
            walls[player] = []

        #Set first real positions
        for player in players:
            if player == my_index:
                area[self.value[0][0], self.value[0][1]] = False
                walls[player].append(self.value[0])
            else:
                area[current_positions[player]] = False
                walls[player].append(current_positions[player])

        #Playout random moves for ennemies after that
        nb_values = len(self.value)
        for index in range(1, nb_values):
            for player in players:
                if player == my_index:
                    area[self.value[index]] = False
                    walls[player].append(self.value[index])
                else:
                    position = self.process_random_position(area, current_positions[player])
                    area[position] = False
                    walls[player].append(position)

        return area, walls

    def process_random_position(self, area, current_position):
        '''
        Return a random position. Position avoid being suicidal, except if there is no other choice
        :param area: state of the area
        :param current_position: current position of the player
        :return: new position
        '''
        offsets = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        possibilities = []
        nb_turns = len(self.value)
        for offset in offsets:
            new_x = current_position[0] + offset[0]
            new_y = current_position[1] + offset[1]
            if 0 <= new_x <= Configuration.MAX_X_GRID and 0 <= new_y <= Configuration.MAX_Y_GRID and area[new_x, new_y]:
                possibilities.append((new_x, new_y))

        if len(possibilities) > 0:
            return possibilities[randint(0, len(possibilities) - 1)]
        else:
            return (1,0) #return nevertheless a valid position

    def play_out(self, context_area, current_positions, players, my_index):
         '''
         Play the simulation
         :return:  True if wins, False otherwise
         '''
         is_game_running = True
         is_win = True
         area, walls = self.initialize_play_out(context_area, current_positions, players, my_index)

         players_game_over = []
         nb_turns_check_separeted = NB_TURNS_CHECK
         turn = 0
         while is_game_running:
            #Start next step of the game
            for player in players:
                position = self.process_random_position(area, current_positions[player])

                if area[position]:
                    area[position] = False
                    walls[player].append(position)
                else:
                    players_game_over.append(player)
                    for wall in walls[player]:
                        area[wall] = True
                    #del self.walls[player] => not mandatory to clean the variables here ...

            for player in players_game_over:
                players.remove(player)
                if player == my_index: #can stop the simulation here
                    is_game_running = False
                    is_win = False

            #TODO: add heuristics Voronoi, tree of chambers, ... to anticipate the end of the game
            #Check the state of the game
            #if nb_turns_check_separeted == 0:
            #   nb_turns_check_separeted = NB_TURNS_CHECK
            #else:
            #   nb_turns_check_separeted -= 1

            if len(players) <= 1:
               is_game_running = False
            else:
                turn += 1
         return is_win, turn

    def expansion(self):
        '''
        Create the next turn of the game
        '''

        last_position = self.value[len(self.value)-1]

        if len(self.value) > 2: previous_position = self.value[len(self.value)-2]
        else: previous_position = None

        offsets = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        for offset in offsets:
            new_x = last_position[0] + offset[0]
            new_y = last_position[1] + offset[1]
            if 0 <= new_x <= Configuration.MAX_X_GRID and 0 <= new_y <= Configuration.MAX_Y_GRID:
                if previous_position is None or (previous_position is not None and new_x != previous_position[0] and new_y != previous_position[1]):
                    self.add_child((new_x, new_y))


def compute_MCTS(area, cur_cycles, list_players, my_index):
    '''
    Compute the MCTS tree from the current situation
    :return: direction to take for this simulation step
    '''

    current_position = cur_cycles[my_index]
    tree = Node(None, current_position)
    tree.expansion()

    nb_iteration = NB_MCTS_ITERATIONS
    while nb_iteration > 0:
        selected_node = tree.selection()
        is_win, nb_turn = selected_node.play_out(area, cur_cycles, list_players, my_index)

        if nb_turn > 0: selected_node.expansion()
        selected_node.back_propagate(is_win)
        nb_iteration -= 1

    max_score = 0
    next_move_to_play = None
    for node in tree.children:
        score = node.score + (A / sqrt(node.number_visit))
        if score > max_score:
            max_score = score
            next_move_to_play = node

    next_position = next_move_to_play.value[-1]
    if next_position[0] - current_position[0] > 0: return 'RIGHT'
    elif next_position[0] - current_position[0] < 0: return 'LEFT'
    elif next_position[1] - current_position[1] > 0: return 'DOWN'
    elif next_position[0] - current_position[0] < 0: return 'UP'
    else: return ''


class MCTSBot():
    def __init__(self):
        self.cur_cycles = {}
        self.wall_cycles = {}
        self.area = ones((30, 20), dtype=bool)
        self.list_players = []
        self.list_players_without_me = []
        self.turn = 0

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

        direction = compute_MCTS(self.area, self.cur_cycles, self.list_players, my_index)
        return direction