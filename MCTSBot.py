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
NB_MCTS_ITERATIONS = 40 #experimental

class Node:
    def __init__(self, parent, value, is_separeted):
        self.score = 0.0
        self.nb_win = 0
        self.number_visit = 1
        self.is_end_game = False
        self.is_always_win = False
        self.children = []
        self.parent = parent
        self.is_separeted = is_separeted

        if parent is None:
            self.depth = 0
            self.value = [value]
        else:
            self.value = self.parent.value[:]
            self.value.append(value)
            self.depth = parent.depth + 1

    def add_child(self, value, is_separeted):
        self.children.append(Node(self, value, is_separeted))

    def back_propagate(self, value, space):
        '''
        Back propagate the result of the simulation in the branch to the root node
        :param value: 1 if it was a win, 0 for a loss
        '''
        #if value:
        #    self.nb_win += 1
        #    self.score = self.nb_win / self.number_visit

        self.score *= self.number_visit

        if value >= (space-value):
            self.score += value/space
        #else:
        #    self.score -= value / space
        self.number_visit += 1

        '''    
        self.score = self.score * self.number_visit
        self.score += (space / 600)
        self.number_visit += 1
        '''
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

    def initialize_play_out(self, context_area, current_positions, previous_moves, players, my_index):
        '''
        Initialize the area with the previous positions of each cycles
        :return: the area at the current state of the game
        :return: the last positions of each player
        '''

        area = numpy.copy(context_area)
        walls = {}

        for player in players:
            # Set past real positions for all players positions
            walls[player] = previous_moves[player][:]

            #Set current real position
            if player == my_index:
                area[self.value[0]] = 0
                walls[player].append(self.value[0])
            else:
                area[current_positions[player]] = 0
                walls[player].append(current_positions[player])

            #Set decided positions for main player and random moves for ennemies
            nb_values = len(self.value)
            for index in range(1, nb_values):
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

    def play_out_by_heuristics(self, context_area, initial_positions, previous_moves, list_players, my_index):

        start = time()
        area, walls = self.initialize_play_out(context_area, initial_positions, previous_moves, list_players, my_index)
        print('initialize play out: ' + str((time()-start)*1000), flush = True)

        current_positions = []
        for i in range(len(list_players)):
            current_positions.append(walls[i][-1])

        # Check if it is an early end game
        if len(list_players) == 2:
            start = time()
            voronoi_area, voronoi_spaces = compute_voronoi_area(area, current_positions, [0,1])
            print('compute voronoi space: ' + str((time() - start) * 1000), flush=True)

            if len(walls[list_players[0]]) >= 10:
                if not self.is_separeted:
                    start = time()
                    distance = compute_path(area, current_positions[my_index], current_positions[1 - my_index])
                    print('compute separation A*: ' + str((time() - start) * 1000), flush=True)
                else:
                    distance = None
                    self.is_separeted = True
                    self.is_end_game = True

                if self.is_separeted:
                    my_articulation_points = detect_articulation_points(area, initial_positions[my_index])
                    ennemy_articulation_points = detect_articulation_points(area, initial_positions[1 - my_index])
                else:
                    start = time()
                    my_articulation_points = detect_articulation_points(area, initial_positions[my_index])
                    print('compute articulation points: ' + str((time() - start) * 1000), flush=True)
                    ennemy_articulation_points = my_articulation_points

                is_in_territory = False
                if self.is_separeted:
                    is_in_territory = len(my_articulation_points) > 0
                else:
                    for articulation in my_articulation_points:
                        if voronoi_area[articulation] == 1:
                            is_in_territory = True
                            break

                if is_in_territory:
                    if len(walls[my_index]) < 2:
                        previous_pos = (-1, -1)
                    else:
                        previous_pos = walls[my_index][-2]

                    start = time()
                    my_spaces = compute_tree_of_chambers(area, voronoi_area, my_articulation_points , current_positions[my_index], previous_pos,1)
                    print('compute my tree of chambers: ' + str((time() - start) * 1000), flush=True)
                else:
                    my_spaces = voronoi_spaces[my_index]

                is_in_territory = False
                if self.is_separeted:
                    is_in_territory = len(ennemy_articulation_points) > 0
                else:
                    for articulation in ennemy_articulation_points:
                        if voronoi_area[articulation] == 2:
                            is_in_territory = True
                            break

                if is_in_territory:
                    if len(walls[1 - my_index]) < 2:
                        previous_pos = (-1, -1)
                    else:
                        previous_pos = walls[1 - my_index][-2]

                    ennemy_spaces = compute_tree_of_chambers(area, voronoi_area, ennemy_articulation_points, current_positions[1-my_index], previous_pos, 2)
                else:
                    ennemy_spaces = voronoi_spaces[1 - my_index]
            else:
                my_spaces = voronoi_spaces[my_index]
                ennemy_spaces = voronoi_spaces[1 - my_index]

            #self.is_always_win = (ennemy_spaces < my_spaces * 2) and distance is None
            return my_spaces, my_spaces + ennemy_spaces

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
                self.add_child((new_x, new_y), self.is_separeted)

def compute_MCTS(area, cur_cycles, previous_moves, list_players, my_index):
    '''
    Compute the MCTS tree from the current situation
    :return: direction to take for this simulation step
    '''

    current_position = cur_cycles[my_index]
    tree = Node(None, current_position, False)
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

            is_win, space = selected_node.play_out_by_heuristics(area, cur_cycles, previous_moves, list_players, my_index)
            if selected_node.is_separeted: selected_node.is_always_win = is_win
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

        if voronoi_area[x, y] > 0:
            sign = 1
        else:
            sign = -1

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
                    if sign == 1:
                        voronoi[0] += 1
                    else:
                        voronoi[1] += 1
                elif next_value == other_neighbor:
                    voronoi_area[new_x, new_y] = Configuration.NEUTRAL_CODE
                    voronoi[Configuration.NEUTRAL_CODE] += 1

    return voronoi

def compute_voronoi_area(area, last_positions, list_players):

    voronoi_area = zeros((Configuration.MAX_X_GRID+1, Configuration.MAX_Y_GRID+1),dtype=int32)
    voronoi_area[last_positions[0]] = list_players[0] + 1
    voronoi_area[last_positions[1]] = list_players[1] + 1

    voronoi = {}
    for i in list_players:
        voronoi[i] = 1
    voronoi[Configuration.NEUTRAL_CODE] = 0

    front_nodes = deque()
    front_nodes.append(last_positions[0])
    front_nodes.append(last_positions[1])

    available_directions = [[0, -1], [0, 1], [-1, 0], [1, 0]]
    while len(front_nodes) > 0:
        cur = front_nodes.popleft()
        x, y = cur[0], cur[1]

        neighbor_value = voronoi_area[x, y]

        if neighbor_value != Configuration.NEUTRAL_CODE:
            for off_x, off_y in available_directions:
                new_x = x + off_x
                new_y = y + off_y

                if 0 <= new_x <= Configuration.MAX_X_GRID and 0 <= new_y <= Configuration.MAX_Y_GRID and not area[new_x, new_y]:
                    next_value = voronoi_area[new_x, new_y]

                    if next_value == 0:
                        voronoi_area[new_x, new_y] = neighbor_value
                        voronoi[neighbor_value-1] += 1
                        front_nodes.append((new_x, new_y))
                    elif next_value != 0 and next_value != neighbor_value:
                        voronoi_area[new_x, new_y] = Configuration.NEUTRAL_CODE
                        voronoi[Configuration.NEUTRAL_CODE] += 1

    return voronoi_area, voronoi

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
    def __init__(self, p_entrance, p_depth, p_parent = None):
        self.space = 0
        self.entrance = p_entrance
        #self.positions = deque() #help in case of merge
        self.parent = p_parent
        self.depth = p_depth
        self.is_leaf = True

def detect_articulation_points(area, root):
    '''
    Find the points that if were filled would separate the board.
    DFS approach
    https://en.wikipedia.org/wiki/Biconnected_component
    :return: list of adjacent points
    '''

    class Node:
        def __init__(self):
            self.depth = None
            self.low = None

    visited_nodes =  zeros((Configuration.MAX_X_GRID+1, Configuration.MAX_Y_GRID+1), dtype=object)
    parents = {}
    available_directions = [[0, -1], [0, 1], [-1, 0], [1, 0]]
    articulations = []

    def f(vertex, p_depth):
        depth = p_depth

        node = Node()
        node.depth = depth
        node.low = depth
        visited_nodes[vertex] = node

        for off_x, off_y in available_directions:
            new_x = vertex[0] + off_x
            new_y = vertex[1] + off_y

            if 0 <= new_x <= Configuration.MAX_X_GRID and 0 <= new_y <= Configuration.MAX_Y_GRID and not area[new_x, new_y]:
                if visited_nodes[new_x, new_y] == 0:
                    parents[(new_x, new_y)] = vertex
                    f((new_x, new_y), p_depth + 1)

                    if vertex != root and visited_nodes[new_x,new_y].low >= visited_nodes[vertex].depth and vertex not in articulations:
                        articulations.append(vertex)

                    visited_nodes[vertex].low = min(visited_nodes[vertex].low, visited_nodes[new_x, new_y].low)
                elif vertex in parents and parents[vertex] != (new_x, new_y):
                    visited_nodes[vertex].low = min(visited_nodes[vertex].low, visited_nodes[new_x, new_y].depth)

    f(root,0)
    return articulations

def compute_tree_of_chambers(area, voronoi_area, articulation_points, current_position, previous_position, voronoi_index_player):
    '''
    Compute the space available with the Tree of chambers algorithm
    '''

    list_chambers = []

    chamber_area = zeros((Configuration.MAX_X_GRID+1, Configuration.MAX_Y_GRID+1), dtype=object) #used as an equivalent of visited_area in BFS algorithm
    front_nodes = deque()

    available_directions = [[0, -1], [0, 1], [-1, 0], [1, 0]]

    #Step 0: Build first chamber and entrance is the step before the current position
    #note that chamber_area[previous_position] == None
    new_chamber = Chamber(previous_position, 0)
    new_chamber.space = 1
    origin_chamber = new_chamber

    chamber_area[current_position] = new_chamber
    front_nodes.append(current_position)

    depth = 1

    #Step 1: Search other chambers
    while len(front_nodes) > 0:
        cur = front_nodes.popleft()
        x, y = cur[0], cur[1]
        current_chamber = chamber_area[cur]

        for off_x, off_y in available_directions:
            new_x = x + off_x
            new_y = y + off_y

            if 0 <= new_x <= Configuration.MAX_X_GRID and 0 <= new_y <= Configuration.MAX_Y_GRID and not area[new_x, new_y]:
                #Step 1-1: if neighbor not in voronoi area of the player => ignore it !
                if voronoi_area[cur] == voronoi_index_player:
                    is_bottle_neck = (new_x, new_y) in articulation_points

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
                        depth += 1
                        new_chamber = Chamber((new_x, new_y), depth, current_chamber)
                        new_chamber.space = 0
                        list_chambers.append(new_chamber)

                        chamber_area[(new_x, new_y)] = new_chamber
                        current_chamber.is_leaf = False
                        front_nodes.append((new_x, new_y))
                    '''
                    else:
                        if chamber_area[(new_x, new_y)] == current_chamber:
                            # Step 1-4: if neighbor associated with the current chamber (or entrance of the current chamber) => ignore it !
                            pass
                        elif chamber_area[(new_x, new_y)] != origin_chamber and current_chamber != origin_chamber and chamber_area[(new_x, new_y)] != current_chamber and chamber_area[(new_x, new_y)] != 0 and current_chamber.entrance != (new_x, new_y):
                            #Step 1-5: if neighbor associated with a chamber different from the current chamber and not the entrance of the current chamber
                            # merge current chamber and neighbor chamber:
                                # identify the lowest common parent chamber
                                # merge the two chambers into the common parent
                            # do NOT add the neighbor to the front queue !
                            pass
                    '''

    #Step 2: Compute spaces between the different leaf chambers and root chamber
    #Step 3: Select best solution (more space => better solution)
    best_space = 0
    for chamber in list_chambers:
        if chamber.is_leaf:
            current_space = 0
            parent = chamber.parent
            new_chamber = chamber
            while parent != origin_chamber:
                current_space += new_chamber.space
                new_chamber = parent
                parent = parent.parent

            if best_space < current_space:
                best_space = current_space

    #No other chambers than the origin one
    best_space += origin_chamber.space

    return best_space

class MCTSBot():
    def __init__(self):
        self.cur_cycles = {}
        self.wall_cycles = {}
        self.area = zeros((30, 20), dtype=int32)
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

        distance = compute_path(self.area, self.wall_cycles[my_index][-1], self.wall_cycles[1 - my_index][-1])

        if distance is None:
            #Play to stick the walls
            direction = 'NORTH'
            best_spaces = 0

            available_directions = [[0, -1], [0, 1], [-1, 0], [1, 0]]
            for off_x, off_y in available_directions:
                new_x = self.cur_cycles[my_index][0] + off_x
                new_y = self.cur_cycles[my_index][1] + off_y

                if 0 <= new_x <= Configuration.MAX_X_GRID and 0 <= new_y <= Configuration.MAX_Y_GRID and not self.area[new_x, new_y]:
                    voronoi_area, voronoi_spaces = compute_voronoi_area(self.area, self.cur_cycles, [0, 1])
                    my_articulation_points = detect_articulation_points(self.area, self.cur_cycles[my_index])
                    my_spaces = compute_tree_of_chambers(self.area, voronoi_area, my_articulation_points, self.wall_cycles[1 - my_index][-1], self.wall_cycles[1 - my_index][-2],my_index+1)

                    if my_spaces > best_spaces:
                        best_spaces = my_spaces

                        if new_x - self.cur_cycles[my_index][0] > 0:
                            direction = 'RIGHT'
                        elif new_x - self.cur_cycles[my_index][0] < 0:
                            direction = 'LEFT'
                        elif new_y - self.cur_cycles[my_index][1] > 0:
                            direction = 'DOWN'
                        elif new_y - self.cur_cycles[my_index][1] < 0:
                            direction = 'UP'
        else:
            direction = compute_MCTS(self.area, self.cur_cycles, self.wall_cycles, self.list_players, my_index)

        print(direction, flush=True)
        self.turn += 1
        return direction