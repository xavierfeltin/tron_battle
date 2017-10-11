import numpy

import Configuration

from math import sqrt, log
from random import randint
from collections import deque
from numpy import ones, zeros, copy, int32
from heapq import heappop, heappush
from time import time

SELECT_CONSTANT = 1.414213 #value from Wikipedia
#SELECT_CONSTANT = 10 #value from paper
NB_TURNS_CHECK = 10 #value from the paper
A = 1 #value from the paper
NB_MCTS_ITERATIONS = 100 #experimental

class Node:
    def __init__(self, parent, value):
        self.score = 0.0
        self.nb_win = 0
        self.number_visit = 1
        self.is_end_game = False
        self.is_always_win = False
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

    def back_propagate(self, value, space):
        '''
        Back propagate the result of the simulation in the branch to the root node
        :param value: 1 if it was a win, 0 for a loss
        '''
        #if value:
        #    self.nb_win += 1
        #    self.score = self.nb_win / self.number_visit

        self.score = self.score * self.number_visit
        self.score += (space / 600)
        self.number_visit += 1
        self.score = self.score / self.number_visit


        if self.parent is not None:
            self.parent.back_propagate(value, space)

    def selection(self):
        '''
        selection of the node that will be extended at this iteration
        :return: Nodes
        '''
        node = self.select_node()
        if len(node.children) > 0:
            node = node.selection()

        return node

    def select_node(self):
        '''
        selection of the next visited nodes for the next extension
        :return: Node
        '''

        max_selection_score = None
        selected_node = None

        if len(self.children) > 0:
            for node in self.children:
                selection_score = node.score + SELECT_CONSTANT * sqrt(log(self.number_visit) / node.number_visit)

                if max_selection_score is None or selection_score > max_selection_score:
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

        #Set first real positions
        for player in players:
            walls[player] = []

            if player == my_index:
                area[self.value[0][0], self.value[0][1]] = Configuration.WALL_CODE
                walls[player].append(self.value[0])
            else:
                area[current_positions[player]] = Configuration.WALL_CODE
                walls[player].append(current_positions[player])

        #Playout random moves for ennemies after that
        nb_values = len(self.value)
        for index in range(1, nb_values):
            for player in players:
                if player == my_index:
                    area[self.value[index]] = Configuration.WALL_CODE
                    walls[player].append(self.value[index])
                else:
                    position = self.process_random_position(area, walls[player][-1])
                    area[position] = Configuration.WALL_CODE
                    walls[player].append(position)

        return area, walls

    def process_random_position(self, area, current_position):
        '''
        Return a random position. Position avoid being suicidal, except if there is no other choice
        :param area: state of the area
        :param current_position: current position of the player
        :return: new position
        '''
        offsets = [[1, 0], [-1, 0], [0, 1], [0, -1]]
        possibilities = []
        nb_turns = len(self.value)
        for x,y in offsets:
            new_x = current_position[0] + x
            new_y = current_position[1] + y
            if 0 <= new_x <= Configuration.MAX_X_GRID and 0 <= new_y <= Configuration.MAX_Y_GRID and area[new_x, new_y] == 0:
                possibilities.append((new_x, new_y))

        if len(possibilities) > 0:
            return possibilities[randint(0, len(possibilities) - 1)]
        else:
            return current_position #return nevertheless a valid position

    def play_out_by_heuristics(self, context_area, initial_positions, list_players, my_index):
        area, walls = self.initialize_play_out(context_area, initial_positions, list_players, my_index)

        current_positions = []
        for i in range(len(list_players)):
            current_positions.append(walls[i][-1])

        # Check if it is an early end game
        if len(list_players) == 2:
            distance = compute_path(area, walls[my_index][-1], walls[1 - my_index][-1])

            if distance is None:  # Two players are separated
                my_spaces, is_separated = process_availables_spaces(area, walls[my_index][-1], [walls[1 - my_index][-1]])
                ennemy_spaces, is_separated = process_availables_spaces(area, walls[1 - my_index][-1], walls[my_index][-1])

                # compute the number of cases for each player
                self.is_end_game = True
                self.is_always_win = (my_spaces - ennemy_spaces) > 0
                return self.is_always_win, my_spaces
            else:
                #Use Voronoi for now
                #voronoi = compute_voronoi(area, current_positions, list_players)

                start = time()
                voronoi = compute_voronoi_bfs(area, current_positions, list_players)
                print('voronoi: ' + str((time()-start)*1000), flush=True)
                space_max = 0
                winner = None
                for player, space in voronoi.items():
                    if space > space_max:
                        winner = player

                return winner == my_index, voronoi[my_index]
        else:
            # Use Voronoi for now
            #voronoi = compute_voronoi(area, current_positions, list_players)

            start = time()
            voronoi = compute_voronoi_bfs(area, current_positions, list_players)
            print('voronoi: ' + str((time() - start) * 1000), flush=True)
            space_max = 0
            winner = None
            for player, space in voronoi.items():
                if space > space_max:
                    winner = player
            return winner == my_index, voronoi[my_index]

    def play_out(self, context_area, initial_positions, list_players, my_index):
         '''
         Play the simulation
         :return:  True if wins, False otherwise
         '''
         is_game_running = True
         is_win = True
         players = list_players[:]
         area, walls = self.initialize_play_out(context_area, initial_positions, players, my_index)

         #Check if it is an early end game
         if len(list_players) == 2:
             my_spaces, is_separated = process_availables_spaces(area, walls[my_index][-1], [walls[1-my_index][-1]])
             if is_separated: #Two players are separated
                ennemy_spaces, is_separated= process_availables_spaces(area, walls[1-my_index][-1],walls[my_index][-1])

                #compute the number of cases for each player
                self.is_end_game = True
                self.is_always_win = (my_spaces - ennemy_spaces) > 0
                return self.is_always_win, 0

         players_game_over = []
         nb_turns_check_separeted = NB_TURNS_CHECK
         turn = 0
         while is_game_running:
            #Start next step of the game
            for player in players:
                position = self.process_random_position(area, walls[player][-1])

                if area[position]:
                    area[position] = Configuration.WALL_CODE
                    walls[player].append(position)
                else:
                    players_game_over.append(player)
                    for wall in walls[player]:
                        area[wall] = 0
                    #del self.walls[player] => not mandatory to clean the variables here ...

            for player in players_game_over:
                players.remove(player)
                players_game_over.remove(player)
                if player == my_index: #can stop the simulation here
                    is_game_running = False
                    is_win = False

            if len(players) <= 1:
               is_game_running = False
            else:
                # TODO: add heuristics Voronoi, tree of chambers, ... to anticipate the end of the game
                # Check the state of the game
                if nb_turns_check_separeted == 0:
                    # Check if it is an early end game
                    if len(list_players) == 2:
                        my_spaces, is_separated = process_availables_spaces(area, walls[my_index][-1], [walls[1 - my_index][-1]])
                        if is_separated:  # Two players are separated
                            ennemy_spaces, is_separated = process_availables_spaces(area, walls[1 - my_index][-1], walls[my_index][-1])

                            # compute the number of cases for each player
                            is_game_running = False
                            is_win = (my_spaces - ennemy_spaces) > 0

                    nb_turns_check_separeted = NB_TURNS_CHECK
                else:
                    nb_turns_check_separeted -= 1

                turn += 1

         return is_win, turn

    def expansion(self, context_area):
        '''
        Create the next turn of the game
        Forbid to add a previous move done by the player in the tree
        and done by all players in previous simulation step
        '''

        last_position = self.value[-1]

        area = numpy.copy(context_area)
        for wall in self.value:
            area[wall] = Configuration.WALL_CODE

        offsets = [[1, 0], [-1, 0], [0, 1], [0, -1]]
        for x,y in offsets:
            new_x = last_position[0] + x
            new_y = last_position[1] + y

            if 0 <= new_x <= Configuration.MAX_X_GRID and 0 <= new_y <= Configuration.MAX_Y_GRID and area[new_x, new_y] == 0:
                self.add_child((new_x, new_y))

def compute_MCTS(area, cur_cycles, list_players, my_index):
    '''
    Compute the MCTS tree from the current situation
    :return: direction to take for this simulation step
    '''

    current_position = cur_cycles[my_index]
    tree = Node(None, current_position)
    tree.expansion(area)

    nb_iteration = NB_MCTS_ITERATIONS
    while nb_iteration > 0:
        selected_node = tree.selection()

        #Test if the game is arleady in a sure win / loss position
        if selected_node.is_end_game:
            selected_node.back_propagate(selected_node.is_always_win, space)
        else:
            #is_win, nb_turn = selected_node.play_out(area, cur_cycles, list_players, my_index)
            #if nb_turn > 0: selected_node.expansion(area)

            is_win, space = selected_node.play_out_by_heuristics(area, cur_cycles, list_players, my_index)
            if not selected_node.is_end_game: selected_node.expansion(area)

        selected_node.back_propagate(is_win, space)
        nb_iteration -= 1

    max_score = 0
    next_move_to_play = None
    for node in tree.children:
        score = node.score + (A / sqrt(node.number_visit))
        if score > max_score:
            max_score = score
            next_move_to_play = node

    if next_move_to_play is not None:
        next_position = next_move_to_play.value[-1]
        if next_position[0] - current_position[0] > 0: return 'RIGHT'
        elif next_position[0] - current_position[0] < 0: return 'LEFT'
        elif next_position[1] - current_position[1] > 0: return 'DOWN'
        elif next_position[1] - current_position[1] < 0: return 'UP'
        else: return ''
    else:
        return ''


def process_availables_spaces(p_area, root, players_position, player_pos = None):
    nb_spaces = 0
    front_nodes = deque()

    area = numpy.copy(p_area)
    area[root] = 0

    if player_pos is not None:
        area[player_pos] = 0

    front_nodes.append(root)
    is_space_shared = True
    available_directions = [[0, -1], [0, 1], [-1, 0], [1, 0]]

    while len(front_nodes) > 0:
        nb_spaces += 1
        cur = front_nodes.popleft()

        for off_x, off_y in available_directions:
            new_x = cur[0] + off_x
            new_y = cur[1] + off_y

            if 0 <= new_x < 30 and 0 <= new_y < 20 and area[new_x, new_y]==0:
                area[new_x, new_y] = Configuration.WALL_CODE
                front_nodes.append((new_x, new_y))

                for position in players_position:
                    is_space_shared = is_space_shared and (position == (new_x, new_y))

    return nb_spaces, is_space_shared

def recursive_flood_fill(p_area, root):
    area = numpy.copy(p_area)
    area[root] = 0

    return fill(p_area, root)

def fill(area, position):
    if area[position]:
        area[position] = Configuration.WALL_CODE

        nb_count = 1
        nb_count += fill(area, (position[0] + 1, position[1]))
        nb_count += fill(area, (position[0] - 1, position[1]))
        nb_count += fill(area, (position[0], position[1]+1))
        nb_count += fill(area, (position[0], position[1]-1))

        return nb_count
    else:
        return 0

def compute_voronoi(area, cycles, list_players):
    voronoi_cells = {}
    for i in list_players:
        voronoi_cells[i] = 0

    for i in range(30):
        for j in range(20):
            if area[i,j] == 0:
                distances = {}
                closest_cycle = None
                is_limit = False

                for k in list_players:
                    distances[k] = abs(i - cycles[k][0]) + abs(j - cycles[k][1])

                    if closest_cycle is None:
                        closest_cycle = k
                    elif distances[k] < distances[closest_cycle]:
                        closest_cycle = k
                    elif distances[k] == distances[closest_cycle]:
                        is_limit = True
                        break

                if not is_limit:
                    voronoi_cells[closest_cycle] += 1

    #msg = ''
    #for player in list_players:
    #    msg += str(player) + ': ' + str(voronoi_cells[player]) + ' '
    #print(str(msg), flush=True)
    return voronoi_cells

def compute_voronoi_bfs(area, last_positions, list_players):
    '''
    Compute voronoi regions based on BFS algorithm.
    It leaves outside the space in closed chambers
    :param area:
    :param last_positions:
    :param list_players:
    :return:
    '''

    voronoi_area = copy(area)
    voronoi_area[last_positions[0]] = 1
    voronoi_area[last_positions[1]] = -1

    voronoi = {}
    voronoi[0] = 1
    voronoi[1] = 1
    voronoi[Configuration.NEUTRAL_CODE] = 0

    front_nodes = deque()
    front_nodes.append(last_positions[0])
    front_nodes.append(last_positions[1])

    available_directions = [[0, -1], [0, 1], [-1, 0], [1, 0]]
    nb_it = 0
    while len(front_nodes) > 0:
        cur = front_nodes.popleft()
        x, y = cur[0], cur[1]

        if voronoi_area[x, y] > 0: sign = 1
        else: sign = -1

        neighbor_value = voronoi_area[x, y] + sign
        other_neighbor = neighbor_value * -1

        for off_x, off_y in available_directions:
            new_x = x + off_x
            new_y = y + off_y

            if 0 <= new_x <= Configuration.MAX_X_GRID and 0 <= new_y <= Configuration.MAX_Y_GRID:
                next_value = voronoi_area[new_x, new_y]

                if next_value == 0:
                    voronoi_area[new_x, new_y] = neighbor_value
                    front_nodes.append((new_x, new_y))
                    if sign == 1: voronoi[0] += 1
                    else: voronoi[1] += 1
                elif next_value == other_neighbor:
                    voronoi_area[new_x, new_y] = Configuration.NEUTRAL_CODE
                    voronoi[Configuration.NEUTRAL_CODE] += 1

    return voronoi

def heuristic(cell, goal):
    '''
    Heuristic for A*, here manhattan distance
    '''
    return abs(cell[0] - goal[0]) + abs(cell[1] - goal[1])


def compute_path(area, root, goal):
    '''
    implementation of A*
    '''

    pr_queue = []
    heappush(pr_queue, (0 + heuristic(root, goal), 0, root))

    visited_area = copy(area)
    visited_area[root] = 0
    available_directions = [[0, -1], [0, 1], [-1, 0], [1, 0]]

    while len(pr_queue) > 0:
        _, cost, current = heappop(pr_queue)  # return the priority in the heap, cost, path and current element

        if current == goal: #Maybe change here to return the element and compute the path after ...
            return cost

        if visited_area[current] == 0:
            visited_area[current] = Configuration.WALL_CODE

            for off_x, off_y in available_directions:
                new_x = current[0] + off_x
                new_y = current[1] + off_y

                if 0 <= new_x < 30 and 0 <= new_y < 20:
                    neighbor = (new_x, new_y)
                    heappush(pr_queue, (cost + heuristic(neighbor, goal), cost + 1, neighbor))
    return None

class Chamber:
    def _init__(self, p_entrance):
        self.space = 0
        self.entrance = p_entrance
        self.contains_battle_front = False
        self.is_leaf = False

def compute_tree_of_chambers(area, voronoi_area, last_positions, walls, player_index):
    '''
    Compute the space available with the Tree of chambers algorithm
    '''

    list_chambers = []
    list_articulation_points = []

    chamber_area = zeros((Configuration.MAX_X_GRID+1, Configuration.MAX_Y_GRID+1), dtype=object)
    visited_area = copy(area)

    front_nodes = deque()

    available_directions = [[0, -1], [0, 1], [-1, 0], [1, 0]]
    nb_spaces = 0

    #Step 0: Build first chamber and entrance is the step before the current position
    #note that chamber_area[previous_position] == None
    new_chamber = Chamber(walls[player_index][-2])
    new_chamber.space = 1
    chamber_area[last_positions[player_index]] = new_chamber
    front_nodes.append(last_positions[player_index])

    #Step 1: Search other chambers
    while len(front_nodes) > 0:
        cur = front_nodes.popleft()
        x, y = cur[0], cur[1]
        current_chamber = chamber_area[cur]

        for off_x, off_y in available_directions:
            new_x = x + off_x
            new_y = y + off_y

            if 0 <= new_x <= Configuration.MAX_X_GRID and 0 <= new_y <= Configuration.MAX_Y_GRID:
                #Step 1-1: if neighbor not in voronoi area of the player => ignore it !
                if numpy.sign(voronoi_area[cur]) > 0:
                    is_bottle_neck = is_articulation_point(area, (new_x, new_y), (off_x*-1, off_y*-1))

                    if chamber_area[(new_x, new_y)] == 0 and not is_bottle_neck:
                        #Step 1-2: if neighbor without chamber and not articulation point:
                            #set current chamber to the neighbor
                            #increment chamber size
                            #add neighbor to the front queue
                        chamber_area[(new_x, new_y)] = current_chamber
                        current_chamber.space += 1
                        front_nodes.append((new_x, new_y))
                    elif chamber_area[(new_x, new_y)] == 0 and is_bottle_neck:
                        #Step 1-3: if neighbor without chamber and is an articulation point:
                            #create a new chamber and affect it to the neighbor
                            #set chamber size to 1
                            #add neighbor to the front queue
                        new_chamber = Chamber((new_x, new_y))
                        new_chamber.space = 1
                        chamber_area[(new_x, new_y)] = new_chamber
                        front_nodes.append((new_x, new_y))
                    else:
                        #Step 1-4: if neighbor associated with the current chamber (or entrance of the current chamber) => ignore it !
                        #Step 1-5: if neighbor associated with a chamber different from the current chamber and not the entrance of the current chamber
                            # merge current chamber and neighbor chamber:
                                # identify the lowest common parent chamber
                                # merge the two chambers into the common parent
                            # do NOT add the neighbor to the front queue !

    #Step 2: Compute spaces between the different leaf chambers and root chamber
    #Step 3: Select best solution (more space => better solution)

                directions = [[0, -1], [0, 1], [-1, 0], [1, 0]]
                directions.remove([off_x, off_y])

                #Detect if the case is an articulation point
                #Improvement: store articulation point information directly when moving players
                free_space = 0
                for off_x2, off_y2 in directions:
                    new_x2 = new_x + off_x2
                    new_y2 = new_y + off_y2
                    if 0 <= new_x2 <= Configuration.MAX_X_GRID and 0 <= new_y2 <= Configuration.MAX_Y_GRID and area[new_x2, new_y2] == 0: free_space+=1

                if free_space == 1:
                    list_articulation_points.append((new_x2, new_y2))
                    new_chamber = Chamber((new_x2, new_y2))

                visited_area[new_x, new_y] = 1

    return nb_spaces

def is_articulation_point(area, position, prev_off):
    '''
    True if the point is an articulation point (only node to connect another part of the graph)
    :param area: current state of the graph
    :param position: position to test
    :param prev_off: previous position coming from to ignore when computing space
    :return: True if it is an articulation point
    '''
    available_directions = [[0, -1], [0, 1], [-1, 0], [1, 0]]
    available_directions.remove(prev_off)

    # Detect if the case is an articulation point
    # Improvement: store articulation point information directly when moving players
    free_space = 0
    for off_x, off_y in available_directions:
        new_x = position[0] + off_x
        new_y = position[1] + off_y
        if 0 <= new_x <= Configuration.MAX_X_GRID and 0 <= new_y <= Configuration.MAX_Y_GRID and area[new_x, new_y] == 0: free_space += 1

    return free_space == 1

class MCTSBot():
    def __init__(self):
        self.cur_cycles = {}
        self.wall_cycles = {}
        self.area = zeros((30, 20), dtype=int32)
        self.list_players = []
        self.list_players_without_me = []
        self.turn = 0

    def compute_direction(self, input):
        self.list_players.clear()
        self.list_players_without_me.clear()

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
                    self.area[x0, y0] = Configuration.WALL_CODE

                self.wall_cycles[i].append((x1, y1))
                self.area[x1, y1] = Configuration.WALL_CODE
            else:
                # If player has lost, remove his wall from the game
                if i in self.cur_cycles:
                    for case in self.wall_cycles[i]:
                        self.area[case] = 0

                    del self.cur_cycles[i]
                    del self.wall_cycles[i]

                    self.list_players.remove(i)
                    self.list_players_without_me.remove(i)

        direction = compute_MCTS(self.area, self.cur_cycles, self.list_players, my_index)
        print(direction, flush=True)
        return direction