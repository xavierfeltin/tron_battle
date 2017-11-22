import sys
from math import sqrt, log
from random import randint
from time import clock
from Utils import compute_path, detect_articulation_points, compute_tree_of_chambers, compute_voronoi, generate_index_cache, generate_manhattan_cache

SELECT_CONSTANT = 1.414213 #value from Wikipedia
#SELECT_CONSTANT = 10 #value from paper
NB_TURNS_CHECK = 10 #value from the paper
A = 1 #value from the paper
NB_MCTS_ITERATIONS = 40 #experimental

class Node:
    def __init__(self, parent, value, is_separeted):
        self.score = -1
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

        if self.score != 1 and self.score != 0:
            if space == 0:
                self.score = 0
            elif space - value <= 1 and value > 1:
                self.score = 1
            else:
                self.score *= self.number_visit
                if value > (space-value):
                    self.score += value/space
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

    def initialize_play_out(self, context_area, current_positions, previous_moves, players, my_index, index_cache):
        '''
        Initialize the area with the previous positions of each cycles
        :return: the area at the current state of the game
        :return: the last positions of each player
        '''

        area = context_area[:]
        walls = {}

        for player in players:
            # Set past real positions for all players positions
            walls[player] = previous_moves[player][:]

            #Set current position
            '''
            if player == my_index:
                area[index_cache[self.value[0][0]][self.value[0][1]]] = False
                walls[player].append(self.value[0])
            else:
                area[index_cache[current_positions[player][0]][current_positions[player][1]]] = False
                walls[player].append(current_positions[player])
            '''

            #Set decided positions for main player and random moves for ennemies
            nb_values = len(self.value)
            for index in range(0, nb_values-1):
                if player == my_index:
                    area[index_cache[self.value[index][0]][self.value[index][1]]] = False
                    walls[player].append(self.value[index])
                else:
                    position = self.process_random_position(area, walls[player][-1], index_cache)
                    area[index_cache[position[0]][position[1]]] = False
                    walls[player].append(position)

            #last position is not registered in the area
            if player == my_index:
                walls[player].append(self.value[nb_values-1])
            else:
                walls[player].append(self.process_random_position(area, walls[player][-1], index_cache))

        return area, walls

    def process_random_position(self, area, current_position, index_cache):
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
            if 0 <= new_x < 30 and 0 <= new_y < 20 and area[index_cache[new_x][new_y]]:
                possibilities.append((new_x, new_y))

        if len(possibilities) > 0:
            return possibilities[randint(0, len(possibilities) - 1)]
        else:
            return current_position #return nevertheless a valid position

    def play_out_by_heuristics(self, context_area, initial_positions, previous_moves, list_players, my_index, manhattan_cache, index_cache):

        start = clock()
        area, walls = self.initialize_play_out(context_area, initial_positions, previous_moves, list_players, my_index, index_cache)
        init_time = (clock() - start) * 1000

        current_positions = []
        for i in range(len(list_players)):
            current_positions.append(walls[i][-1])

        msg = ''
        # Check if it is an early end game
        if len(list_players) == 2:
            start = clock()
            voronoi_area, voronoi_spaces = compute_voronoi(area, current_positions, [0,1], index_cache)
            voronoi_time = (clock() - start) * 1000
            #msg += 'voronoi: ' + str(round(voronoi_time,2)) + 'ms'

            if len(walls[list_players[0]]) >= 10 and voronoi_spaces[0] != 0 and voronoi_spaces[1] != 0:
                if not self.is_separeted:
                    r_x,r_y = current_positions[my_index][0], current_positions[my_index][1]
                    g_x,g_y = current_positions[1 - my_index][0], current_positions[1 - my_index][1]

                    #start = clock()
                    #distance = compute_path(area, current_positions[my_index], index_cache[r_x][r_y],current_positions[1 - my_index], index_cache[g_x][g_y], manhattan_cache, index_cache)
                    #path_time = (clock() - start) * 1000
                    #msg += ', path: ' + str(round(path_time,2)) + 'ms'

                    if voronoi_spaces[2] == 0:
                        self.is_separeted = True
                        #self.is_end_game = True

                if self.is_separeted:
                    start = clock()
                    my_articulation_points = detect_articulation_points(area, initial_positions[my_index], index_cache[initial_positions[my_index][0]][initial_positions[my_index][1]], index_cache)
                    ennemy_articulation_points = detect_articulation_points(area, initial_positions[1 - my_index], index_cache[initial_positions[1 - my_index][0]][initial_positions[1 - my_index][1]], index_cache)
                    articulation_separated_time = (clock() - start) * 1000
                    #msg += ', AP sep: ' + str(round(articulation_separated_time,2)) + 'ms'
                else:
                    start = clock()
                    my_articulation_points = detect_articulation_points(area, initial_positions[my_index], index_cache[initial_positions[my_index][0]][initial_positions[my_index][1]],index_cache)
                    ennemy_articulation_points = my_articulation_points
                    articulation_time = (clock() - start) * 1000
                    #msg += ', AP: ' + str(round(articulation_time,2)) + 'ms'

                is_in_territory = False
                if self.is_separeted:
                    is_in_territory = len(my_articulation_points) > 0
                else:
                    for articulation in my_articulation_points:
                        if voronoi_area[articulation] == my_index:
                            is_in_territory = True
                            break

                if is_in_territory:
                    if len(walls[my_index]) < 2:
                        previous_pos = (-1, -1)
                    else:
                        previous_pos = walls[my_index][-2]
                    start = clock()
                    my_spaces = compute_tree_of_chambers(area, voronoi_area, my_articulation_points , current_positions[my_index], previous_pos,index_cache, my_index)
                    tree_time = (clock()-start) * 1000
                    #msg += ', My tree: ' + str(round(tree_time,2)) + 'ms'
                else:
                    my_spaces = voronoi_spaces[my_index]

                is_in_territory = False
                if self.is_separeted:
                    is_in_territory = len(ennemy_articulation_points) > 0
                else:
                    for articulation in ennemy_articulation_points:
                        if voronoi_area[articulation] == (1-my_index):
                            is_in_territory = True
                            break

                if is_in_territory:
                    if len(walls[1 - my_index]) < 2:
                        enn_previous_pos = (-1, -1)
                    else:
                        enn_previous_pos = walls[1 - my_index][-2]

                    start = clock()
                    ennemy_spaces = compute_tree_of_chambers(area, voronoi_area, ennemy_articulation_points, current_positions[1-my_index], enn_previous_pos,index_cache, (1-my_index))
                    tree_time = (clock() - start) * 1000
                    #msg += ', Ennemy tree: ' + str(round(tree_time,2)) + 'ms'
                else:
                    ennemy_spaces = voronoi_spaces[1 - my_index]
            else:
                my_spaces = voronoi_spaces[my_index]
                ennemy_spaces = voronoi_spaces[1 - my_index]

            nb_walls = 0
            for i in walls:
                nb_walls += len(walls[i])

            msg += ', my: ' + str(voronoi_spaces[0])+'/'+str(my_spaces) + ', enn: ' + str(voronoi_spaces[1])+'/'+str(ennemy_spaces) + ', null: ' + str(voronoi_spaces[2]) + ', wall: ' + str(nb_walls)
            msg += ', total: ' + str(my_spaces+ennemy_spaces+voronoi_spaces[2]+nb_walls)

            print(msg, flush=True)
            return my_spaces, my_spaces + ennemy_spaces

        else:
            # Use Voronoi for now
            voronoi_area, voronoi_spaces = compute_voronoi(area, current_positions, list_players, index_cache)

            space_max = 0
            winner = None
            for player, space in voronoi_spaces.items():
                if space > space_max:
                    winner = player
            return winner == my_index, voronoi_spaces[my_index]

    def expansion(self, context_area, index_cache):
        '''
        Create the next turn of the game
        Forbid to add a previous move done by the player in the tree
        and done by all players in previous simulation step
        '''

        last_position = self.value[-1]

        area = context_area[:]
        for wall in self.value:
            area[index_cache[wall[0]][wall[1]]] = False

        offsets = [[1, 0], [-1, 0], [0, 1], [0, -1]]
        for x,y in offsets:
            new_x = last_position[0] + x
            new_y = last_position[1] + y

            if 0 <= new_x < 30 and 0 <= new_y < 20 and area[index_cache[new_x][new_y]]:
                self.add_child((new_x, new_y), self.is_separeted)

def compute_MCTS(area, cur_cycles, previous_moves, list_players, my_index, manhattan_cache, index_cache):
    '''
    Compute the MCTS tree from the current situation
    :return: direction to take for this simulation step
    '''

    current_position = cur_cycles[my_index]
    tree = Node(None, current_position, False)
    tree.expansion(area,index_cache)

    nb_iteration = NB_MCTS_ITERATIONS
    while nb_iteration > 0:
        selected_node = tree.selection()

        #Test if the game is arleady in a sure win / loss position
        #if selected_node.is_end_game:
        #    selected_node.back_propagate(selected_node.is_always_win, space)
        #else:
        my_space, space = selected_node.play_out_by_heuristics(area, cur_cycles, previous_moves, list_players, my_index, manhattan_cache, index_cache)
        if selected_node.is_separeted: selected_node.is_always_win = my_space
        if not selected_node.is_end_game: selected_node.expansion(area,index_cache)

        selected_node.back_propagate(my_space, space)
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

class MCTSBot():
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

        direction = compute_MCTS(self.area, self.current_move, self.wall_cycles, self.list_players, my_index, self.manhattan_cache, self.index_cache)

        print(direction, flush=True)
        self.turn += 1
        return direction