import sys
from math import sqrt, log
from random import randint
from collections import deque
from heapq import heappop, heappush
from time import clock

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
            voronoi_area, voronoi_spaces = compute_voronoi(area, current_positions, [0,1],manhattan_cache, index_cache)
            voronoi_time = (clock() - start) * 1000
            msg += 'voronoi: ' + str(round(voronoi_time,2)) + 'ms'

            if len(walls[list_players[0]]) >= 10:
                if not self.is_separeted:
                    r_x,r_y = current_positions[my_index][0], current_positions[my_index][1]
                    g_x,g_y = current_positions[1 - my_index][0], current_positions[1 - my_index][1]

                    start = clock()
                    distance = compute_path(area, current_positions[my_index], index_cache[r_x][r_y],current_positions[1 - my_index], index_cache[g_x][g_y], manhattan_cache, index_cache)
                    path_time = (clock() - start) * 1000
                    msg += ', path: ' + str(round(path_time,2)) + 'ms'

                    if distance is None:
                        self.is_separeted = True
                        #self.is_end_game = True

                if self.is_separeted:
                    start = clock()
                    my_articulation_points = detect_articulation_points(area, initial_positions[my_index], index_cache[initial_positions[my_index][0]][initial_positions[my_index][1]], index_cache)
                    ennemy_articulation_points = detect_articulation_points(area, initial_positions[1 - my_index], index_cache[initial_positions[1 - my_index][0]][initial_positions[1 - my_index][1]], index_cache)
                    articulation_separated_time = (clock() - start) * 1000
                    msg += ', AP sep: ' + str(round(articulation_separated_time,2)) + 'ms'
                else:
                    start = clock()
                    my_articulation_points = detect_articulation_points(area, initial_positions[my_index], index_cache[initial_positions[my_index][0]][initial_positions[my_index][1]],index_cache)
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
                    if len(walls[my_index]) < 2:
                        previous_pos = (-1, -1)
                    else:
                        previous_pos = walls[my_index][-2]
                    start = clock()
                    my_spaces = compute_tree_of_chambers(area, voronoi_area, my_articulation_points , current_positions[my_index], previous_pos,index_cache, my_index)
                    tree_time = (clock()-start) * 1000
                    msg += ', My tree: ' + str(round(tree_time,2)) + 'ms'
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
                    msg += ', Ennemy tree: ' + str(round(tree_time,2)) + 'ms'
                else:
                    ennemy_spaces = voronoi_spaces[1 - my_index]
            else:
                my_spaces = voronoi_spaces[my_index]
                ennemy_spaces = voronoi_spaces[1 - my_index]


            nb_walls = 0
            for i in walls:
                nb_walls += len(walls[i])

            msg += ', my: ' + str(my_spaces) + ', enn: ' + str(ennemy_spaces) + ', null: ' + str(voronoi_spaces[2]) + ', wall: ' + str(nb_walls)
            msg += ', total: ' + str(my_spaces+ennemy_spaces+voronoi_spaces[2]+nb_walls)

            if my_spaces+ennemy_spaces+voronoi_spaces[2]+nb_walls < 500:
                msg2 = 'depth: ' + str(self.depth) +  ', my: ' + str(my_spaces) + ', enn: ' + str(ennemy_spaces) + ', null: ' + str(
                    voronoi_spaces[2]) + ', wall: ' + str(nb_walls)
                msg2 += ', total: ' + str(my_spaces + ennemy_spaces + voronoi_spaces[2] + nb_walls)
                msg2 += '\n'
                msg2 += 'me: ' + str(current_positions[my_index]) +', ennemy: ' + str(current_positions[1-my_index])
                msg2 += '\n'
                for i in range(30):
                    msg2 += ' '
                    for j in range(20):
                        msg2 += str(area[index_cache[i][j]]) + ' '
                    msg2 += '\n'
                print(msg2, file=sys.stderr, flush=True)
                print('-----------------', file=sys.stderr, flush=True)
            print(msg, flush=True)


            area[index_cache[walls[my_index][-1][0]][walls[my_index][-1][1]]] = False
            area[index_cache[walls[1-my_index][-1][0]][walls[1-my_index][-1][1]]] = False

            return my_spaces, my_spaces + ennemy_spaces

        else:
            # Use Voronoi for now
            voronoi_area, voronoi_spaces = compute_voronoi(area, current_positions, list_players, manhattan_cache, index_cache)

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
        if selected_node.is_end_game:
            selected_node.back_propagate(selected_node.is_always_win, space)
        else:
            is_win, space = selected_node.play_out_by_heuristics(area, cur_cycles, previous_moves, list_players, my_index, manhattan_cache, index_cache)
            if selected_node.is_separeted: selected_node.is_always_win = is_win
            if not selected_node.is_end_game: selected_node.expansion(area,index_cache)

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

def compute_voronoi(area, last_positions, list_players, manhattan_cache, index_cache):

    voronoi_area = [-1] * 600
    neutral_index = len(list_players)
    voronoi = [0] * (neutral_index + 1)

    for x in range(30):
        for y in range(20):
            if area[index_cache[x][y]]:
                min_distance = 100
                min_player = -1
                nb_players = 0

                for player in list_players:
                    xp, yp = last_positions[player][0], last_positions[player][1]
                    distance = manhattan_cache[x][y][xp][yp]

                    if distance < min_distance:
                        min_distance = distance
                        min_player = player
                        nb_players = 1
                    elif distance == min_distance:
                        nb_players += 1

                if nb_players == 1:
                    voronoi_area[index_cache[x][y]] = min_player
                    voronoi[min_player] += 1
                else:
                    voronoi[neutral_index] += 1

    return voronoi_area, voronoi

def compute_path(area, root, r_index, goal, g_index, cache_manhattan, cache_index):
    '''
    implementation of A*
    '''

    x_goal = goal[0]
    y_goal = goal[1]

    pr_queue = []
    heappush(pr_queue, (0 + cache_manhattan[root[0]][root[1]][x_goal][y_goal], 0, root))

    to_visit_area = area[:]

    to_visit_area[r_index] = True
    to_visit_area[g_index] = True
    available_directions = [[0, -1], [0, 1], [-1, 0], [1, 0]]

    while pr_queue:
        _, cost, current = heappop(pr_queue)  # return the priority in the heap, cost, path and current element
        x = current[0]
        y = current[1]
        cur_index = cache_index[x][y]

        if cur_index == g_index: #Maybe change here to return the element and compute the path after ...
            return cost

        if to_visit_area[cur_index]:
            to_visit_area[cur_index] = False

            for off_x, off_y in available_directions:
                new_x = x + off_x
                new_y = y + off_y

                if 0 <= new_x < 30 and 0 <= new_y < 20 and to_visit_area[cache_index[new_x][new_y]]:
                    heappush(pr_queue, (cost + cache_manhattan[new_x][new_y][x_goal][y_goal], cost + 1, (new_x, new_y)))

    return None

class Chamber:
    def __init__(self, p_entrance, p_depth, p_parent = None):
        self.space = 0
        self.entrance = p_entrance
        #self.positions = deque() #help in case of merge
        self.parent = p_parent
        self.depth = p_depth
        self.is_leaf = True

def detect_articulation_points(area, root, r_index, index_cache):
    '''
    Find the points that if were filled would separate the board.
    DFS approach
    https://en.wikipedia.org/wiki/Biconnected_component
    :return: list of adjacent points
    '''

    to_visit = [True] * 600
    a_low = [0] * 600
    a_depth = [0] * 600

    parents = {}
    available_directions = [[0, -1], [0, 1], [-1, 0], [1, 0]]
    articulations = []

    def f(vertex, v_index, p_depth):
        a_low[v_index] = p_depth
        a_depth[v_index] = p_depth
        to_visit[v_index] = False

        for off_x, off_y in available_directions:
            new_x = vertex[0] + off_x
            new_y = vertex[1] + off_y

            if 0 <= new_x < 30 and 0 <= new_y < 20 and area[index_cache[new_x][new_y]]:
                new_index = index_cache[new_x][new_y]
                if to_visit[new_index]:
                    parents[new_index] = v_index
                    f((new_x, new_y), new_index, p_depth + 1)

                    if v_index != r_index and a_low[new_index] >= a_depth[v_index] and v_index not in articulations:
                        articulations.append(v_index)

                    a_low[v_index] = min(a_low[v_index], a_low[new_index])
                elif v_index in parents and parents[v_index] != new_index:
                    a_low[v_index] = min(a_low[v_index], a_depth[new_index])

    f(root,r_index,0)

    return articulations

def compute_tree_of_chambers(area, voronoi_area, articulation_points, current_position, previous_position, index_cache, player_index):
    '''
    Compute the space available with the Tree of chambers algorithm
    '''

    list_chambers = []

    chamber_area = [None] * 600  # used as an equivalent of visited_area in BFS algorithm

    front_nodes = deque()
    available_directions = [[0, -1], [0, 1], [-1, 0], [1, 0]]

    # Step 0: Build first chamber and entrance is the step before the current position
    # note that chamber_area[previous_position] == None
    new_chamber = Chamber(previous_position, 0)
    new_chamber.space = 1
    origin_chamber = new_chamber

    chamber_area[index_cache[current_position[0]][current_position[1]]] = new_chamber
    front_nodes.append(current_position)

    depth = 1

    # Step 1: Search other chambers
    while front_nodes:
        cur = front_nodes.popleft()
        x, y = cur[0], cur[1]
        current_index = index_cache[x][y]
        current_chamber = chamber_area[current_index]

        for off_x, off_y in available_directions:
            new_x = x + off_x
            new_y = y + off_y

            if 0 <= new_x < 30 and 0 <= new_y < 20 and area[index_cache[new_x][new_y]]:
                new_index = index_cache[new_x][new_y]

                # Step 1-1: if neighbor not in voronoi area of the player => ignore it !
                if voronoi_area[new_index] == player_index:
                    is_bottle_neck = new_index in articulation_points

                    if chamber_area[new_index] is None and not is_bottle_neck:
                        # Step 1-2: if neighbor without chamber and not articulation point:
                        # set current chamber to the neighbor
                        # increment chamber size
                        # add neighbor to the front queue
                        chamber_area[new_index] = current_chamber
                        current_chamber.space += 1
                        front_nodes.append((new_x, new_y))
                    elif chamber_area[new_index] is None and is_bottle_neck:
                        # Step 1-3: if neighbor without chamber and is an articulation point:
                        # create a new chamber and affect it to the neighbor
                        # set chamber size to 1
                        # add neighbor to the front queue
                        depth += 1
                        new_chamber = Chamber((new_x, new_y), depth, current_chamber)
                        new_chamber.space = 0
                        list_chambers.append(new_chamber)

                        chamber_area[new_index] = new_chamber
                        current_chamber.is_leaf = False
                        front_nodes.append((new_x, new_y))

    # Step 2: Compute spaces between the different leaf chambers and root chamber
    # Step 3: Select best solution (more space => better solution)
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

    # No other chambers than the origin one
    best_space += origin_chamber.space

    return best_space

def generate_manhattan_cache():
    '''
    Cache to store all the manhattan distance between the different grid cells
    '''
    manhattan_cache = [None] * 30
    for x1 in range(30):
        for y1 in range(20):
            for x2 in range(30):
                for y2 in range(20):
                    if manhattan_cache[x1] is None:
                        manhattan_cache[x1] = [None] * 20
                    if manhattan_cache[x1][y1] is None:
                        manhattan_cache[x1][y1] = [None] * 30
                    if manhattan_cache[x1][y1][x2] is None:
                        manhattan_cache[x1][y1][x2] = [None] * 20
                    if manhattan_cache[x1][y1][x2][y2] is None:
                        manhattan_cache[x1][y1][x2][y2] = [0] * 30

                    manhattan_cache[x1][y1][x2][y2] = abs(x1 - x2) + abs(y1 - y2)
    return manhattan_cache

def generate_index_cache():
    '''
    Cache to store all the linearized indexes of the grid
    '''
    index_cache = [None] * 30
    for x in range(30):
        for y in range(20):
            if index_cache[x] is None:
                index_cache[x] = [0] * 20

            index_cache[x][y] = x + 30 * y
    return index_cache

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

        x_root, y_root = self.current_move[my_index][0], self.current_move[my_index][1]
        x_goal, y_goal =  self.current_move[1 - my_index][0], self.current_move[1 - my_index][1]
        distance = compute_path(self.area, self.current_move[my_index], self.index_cache[x_root][y_root], self.current_move[1 - my_index], self.index_cache[x_goal][y_goal], self.manhattan_cache, self.index_cache)

        if distance is None:
            #Play to stick the walls when players are separated
            direction = 'NORTH'
            best_spaces = 0

            available_directions = [[0, -1], [0, 1], [-1, 0], [1, 0]]
            for off_x, off_y in available_directions:
                new_x = self.current_move[my_index][0] + off_x
                new_y = self.current_move[my_index][1] + off_y

                if 0 <= new_x < 30 and 0 <= new_y < 20 and self.area[self.index_cache[new_x][new_y]]:
                    voronoi_area, voronoi_spaces = compute_voronoi(self.area, self.current_move, [0, 1], self.manhattan_cache, self.index_cache)

                    my_articulation_points = detect_articulation_points(self.area, self.current_move[my_index], self.index_cache[self.current_move[my_index][0]][self.current_move[my_index][1]], self.index_cache)

                    my_spaces = compute_tree_of_chambers(self.area, voronoi_area, my_articulation_points, self.current_move[1 - my_index], self.previous_move[1 - my_index],self.index_cache, my_index)

                    if my_spaces > best_spaces:
                        best_spaces = my_spaces

                        if new_x - self.current_move[my_index][0] > 0:
                            direction = 'RIGHT'
                        elif new_x - self.current_move[my_index][0] < 0:
                            direction = 'LEFT'
                        elif new_y - self.current_move[my_index][1] > 0:
                            direction = 'DOWN'
                        elif new_y - self.current_move[my_index][1] < 0:
                            direction = 'UP'
        else:
            direction = compute_MCTS(self.area, self.current_move, self.wall_cycles, self.list_players, my_index, self.manhattan_cache, self.index_cache)

        print(direction, flush=True)
        self.turn += 1
        return direction