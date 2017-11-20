import Configuration
import numpy
import sys
from numpy import ones, zeros, copy, int32, array
from collections import deque
from time import clock
from heapq import heappop, heappush

class Chamber:
    def __init__(self, p_entrance, p_depth, p_parent = None):
        self.space = 0
        self.entrance = p_entrance
        self.positions = deque() #help in case of merge
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

def detect_articulation_points_array(area, root, r_index):
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

    visited_nodes = zeros(600, dtype=object)

    parents = {}
    available_directions = [[0, -1], [0, 1], [-1, 0], [1, 0]]
    articulations = []

    def f(vertex, v_index, p_depth):
        depth = p_depth

        node = Node()
        node.depth = depth
        node.low = depth
        visited_nodes[v_index] = node

        for off_x, off_y in available_directions:
            new_x = vertex[0] + off_x
            new_y = vertex[1] + off_y

            new_index = new_y * 30 + new_x

            if 0 <= new_x <= Configuration.MAX_X_GRID and 0 <= new_y <= Configuration.MAX_Y_GRID and not area[new_index]:
                new_node = visited_nodes[new_index]
                if new_node == 0:
                    parents[new_index] = v_index
                    f((new_x, new_y), new_index, p_depth + 1)
                    new_node = visited_nodes[new_index]

                    if v_index != r_index and new_node.low >= node.depth and v_index not in articulations:
                        articulations.append(v_index)

                    node.low = min(node.low, new_node.low)
                elif v_index in parents and parents[v_index] != new_index:
                    node.low = min(node.low, new_node.depth)

    f(root,r_index,0)

    return articulations

def detect_articulation_points_array_without_nodes(area, root, r_index):
    '''
    Find the points that if were filled would separate the board.
    DFS approach
    https://en.wikipedia.org/wiki/Biconnected_component
    :return: list of adjacent points
    '''

    visited_nodes = zeros(600, dtype=bool)
    a_low = zeros(600, dtype=int32)
    a_depth = zeros(600, dtype=int32)

    parents = {}
    available_directions = [[0, -1], [0, 1], [-1, 0], [1, 0]]
    articulations = []

    def f(vertex, v_index, p_depth):
        depth = p_depth

        a_low[v_index] = p_depth
        a_depth[v_index] = p_depth
        visited_nodes[v_index] = True

        for off_x, off_y in available_directions:
            new_x = vertex[0] + off_x
            new_y = vertex[1] + off_y

            new_index = new_y * 30 + new_x

            if 0 <= new_x <= Configuration.MAX_X_GRID and 0 <= new_y <= Configuration.MAX_Y_GRID and not area[new_index]:
                if not visited_nodes[new_index]:
                    parents[new_index] = v_index
                    f((new_x, new_y), new_index, p_depth + 1)

                    if v_index != r_index and a_low[new_index] >= a_depth[v_index] and v_index not in articulations:
                        articulations.append(v_index)

                    a_low[v_index] = min(a_low[v_index], a_low[new_index])
                elif v_index in parents and parents[v_index] != new_index:
                    a_low[v_index] = min(a_low[v_index], a_depth[new_index])

    f(root,r_index,0)

    return articulations

def detect_articulation_points_array_without_nodes_with_array(area, root, r_index, index_cache):
    '''
    Find the points that if were filled would separate the board.
    DFS approach
    https://en.wikipedia.org/wiki/Biconnected_component
    :return: list of adjacent points
    '''

    to_visit = [True] * 600
    a_low = [0] * 600
    a_depth = [0] * 600

    #parents = {}
    parents = [-1] * 600
    available_directions = [[0, -1], [0, 1], [-1, 0], [1, 0]]
    articulations = set()

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
                #elif v_index in parents and parents[v_index] != new_index:
                elif parents[v_index] != -1 and parents[v_index] != new_index:
                    a_low[v_index] = min(a_low[v_index], a_depth[new_index])

    f(root,r_index,0)

    return articulations

def compute_tree_of_chambers(area, voronoi_area, articulation_points, current_position, previous_position):
    '''
    Compute the space available with the Tree of chambers algorithm
    '''

    list_chambers = []

    chamber_area = zeros((Configuration.MAX_X_GRID+1, Configuration.MAX_Y_GRID+1), dtype=object) #used as an equivalent of visited_area in BFS algorithm
    front_nodes = deque()

    available_directions = [[0, -1], [0, 1], [-1, 0], [1, 0]]
    nb_spaces = 0

    #Step 0: Build first chamber and entrance is the step before the current position
    #note that chamber_area[previous_position] == None
    new_chamber = Chamber(previous_position, 0)
    new_chamber.space = 1
    origin_chamber = new_chamber

    chamber_area[current_position] = new_chamber
    new_chamber.positions.append(current_position)
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
                if numpy.sign(voronoi_area[cur]) > 0:
                    is_bottle_neck = (new_x, new_y) in articulation_points

                    if chamber_area[(new_x, new_y)] == 0 and not is_bottle_neck:
                        #Step 1-2: if neighbor without chamber and not articulation point:
                            #set current chamber to the neighbor
                            #increment chamber size
                            #add neighbor to the front queue
                        chamber_area[(new_x, new_y)] = current_chamber
                        current_chamber.space += 1
                        current_chamber.positions.append((new_x, new_y))
                        front_nodes.append((new_x, new_y))
                    elif chamber_area[(new_x, new_y)] == 0 and is_bottle_neck:
                        #Step 1-3: if neighbor without chamber and is an articulation point:
                            #create a new chamber and affect it to the neighbor
                            #set chamber size to 1
                            #add neighbor to the front queue
                        depth += 1
                        new_chamber = Chamber((new_x, new_y), depth, current_chamber)
                        new_chamber.space = 0
                        new_chamber.positions.append((new_x, new_y))
                        list_chambers.append(new_chamber)

                        chamber_area[(new_x, new_y)] = new_chamber
                        current_chamber.is_leaf = False
                        front_nodes.append((new_x, new_y))

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

def compute_tree_of_chambers_optimized(area, voronoi_area, articulation_points, current_position, previous_position, index_cache, player_index):
    '''
    Compute the space available with the Tree of chambers algorithm
    '''

    list_chambers = []

    chamber_area = [None] * 600 #used as an equivalent of visited_area in BFS algorithm
    #chamber_area = zeros((Configuration.MAX_X_GRID+1, Configuration.MAX_Y_GRID+1), dtype=object)

    front_nodes = deque()

    available_directions = [[0, -1], [0, 1], [-1, 0], [1, 0]]
    nb_spaces = 0

    #Step 0: Build first chamber and entrance is the step before the current position
    #note that chamber_area[previous_position] == None
    new_chamber = Chamber(previous_position, 0)
    new_chamber.space = 1
    origin_chamber = new_chamber

    chamber_area[index_cache[current_position[0]][current_position[1]]] = new_chamber
    new_chamber.positions.append(current_position)
    front_nodes.append(current_position)

    depth = 1

    #Step 1: Search other chambers
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

                #Step 1-1: if neighbor not in voronoi area of the player => ignore it !
                if voronoi_area[current_index] == player_index:
                    is_bottle_neck = new_index in articulation_points

                    if chamber_area[new_index] is None and not is_bottle_neck:
                        #Step 1-2: if neighbor without chamber and not articulation point:
                            #set current chamber to the neighbor
                            #increment chamber size
                            #add neighbor to the front queue
                        chamber_area[new_index] = current_chamber
                        current_chamber.space += 1
                        current_chamber.positions.append((new_x, new_y))
                        front_nodes.append((new_x, new_y))
                    elif chamber_area[new_index] is None and is_bottle_neck:
                        #Step 1-3: if neighbor without chamber and is an articulation point:
                            #create a new chamber and affect it to the neighbor
                            #set chamber size to 1
                            #add neighbor to the front queue
                        depth += 1
                        new_chamber = Chamber((new_x, new_y), depth, current_chamber)
                        new_chamber.space = 0
                        new_chamber.positions.append((new_x, new_y))
                        list_chambers.append(new_chamber)

                        chamber_area[new_index] = new_chamber
                        current_chamber.is_leaf = False
                        front_nodes.append((new_x, new_y))

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

def compute_voronoi_area_without_numpy(area, last_positions, list_players, index_cache):

    to_visit_area = area[:]
    to_visit_area[index_cache[last_positions[0][0]][last_positions[0][1]]] = True
    to_visit_area[index_cache[last_positions[1][0]][last_positions[1][1]]] = True

    voronoi_area = [-1] * 600
    voronoi_area[index_cache[last_positions[0][0]][last_positions[0][1]]] = list_players[0]
    voronoi_area[index_cache[last_positions[1][0]][last_positions[1][1]]] = list_players[1]

    neutral_index = len(list_players)
    voronoi = [1] * (neutral_index+1)
    voronoi[neutral_index] = 0

    front_nodes = deque()
    front_nodes.append(last_positions[0])
    front_nodes.append(last_positions[1])

    available_directions = [[0, -1], [0, 1], [-1, 0], [1, 0]]
    while front_nodes:
        cur = front_nodes.popleft()
        x, y = cur[0], cur[1]
        front_index = index_cache[x][y]

        neighbor_value = voronoi_area[front_index]

        #if neighbor_value != Configuration.NEUTRAL_CODE:
        for off_x, off_y in available_directions:
            new_x = x + off_x
            new_y = y + off_y

            if 0 <= new_x < 30 and 0 <= new_y < 20 and to_visit_area[index_cache[new_x][new_y]]:
                new_index = index_cache[new_x][new_y]

                to_visit_area[new_index] = False
                next_value = voronoi_area[new_index]

                if next_value == -1:
                    voronoi_area[new_index] = neighbor_value
                    voronoi[neighbor_value] += 1
                    front_nodes.append((new_x, new_y))
                elif next_value != -1 and next_value != neighbor_value:
                    voronoi_area[new_index] = Configuration.NEUTRAL_CODE
                    voronoi[neutral_index] += 1

    return voronoi_area, voronoi

def compute_voronoi_area_without_numpy2(area, last_positions, list_players, index_cache):
    voronoi_area = [-1] * 600
    front_nodes = deque()
    voronoi = {}
    neutral_index = len(list_players)

    p_indexes = {}

    for player in list_players:
        p_pos = last_positions[player]
        p_indexes[player] = index_cache[p_pos[0]][p_pos[1]]
        voronoi_area[p_indexes[player]] = player
        front_nodes.append(p_pos)
        voronoi[player] = 1
    voronoi[neutral_index] = 0

    available_directions = [[0, -1], [0, 1], [-1, 0], [1, 0]]

    while front_nodes:
        cur = front_nodes.popleft()
        x, y = cur[0], cur[1]
        front_index = index_cache[x][y]
        neighbor_value = voronoi_area[front_index]

        #if neighbor_value != neutral_index:
        for off_x, off_y in available_directions:
            new_x = x + off_x
            new_y = y + off_y

            if 0 <= new_x < 30 and 0 <= new_y < 20 and area[index_cache[new_x][new_y]]:
                new_index = index_cache[new_x][new_y]
                next_value = voronoi_area[new_index]

                if next_value == -1:
                    voronoi_area[new_index] = neighbor_value
                    voronoi[neighbor_value] += 1
                    front_nodes.append((new_x, new_y))
                elif next_value != neighbor_value:
                    voronoi_area[new_index] = neutral_index
                    voronoi[neutral_index] += 1

    return voronoi_area, voronoi

def compute_voronoi_area_with_manhattan(area, last_positions, list_players, manhattan_cache, index_cache):

    voronoi_area = [-1] * 600
    neutral_index = len(list_players)
    voronoi = [1] * (neutral_index + 1)
    voronoi[neutral_index] = 0

    for x in range(30):
        for y in range(20):
            min_distance = 100
            min_player = []

            if area[index_cache[x][y]]:
                for player in list_players:
                    xp, yp = last_positions[player][0], last_positions[player][1]
                    distance = manhattan_cache[x][y][xp][yp]

                    if distance < min_distance:
                        min_distance = distance
                        min_player = [player]
                    elif distance == min_distance:
                        min_player.append(player)

                if len(min_player) == 1:
                    voronoi_area[index_cache[x][y]] = min_player[0]
                    voronoi[min_player[0]] += 1
                else:
                    voronoi_area[index_cache[x][y]] = Configuration.NEUTRAL_CODE
                    voronoi[neutral_index] += 1

    return voronoi_area, voronoi

def compute_voronoi_with_a_star(area, last_positions, list_players, manhattan_cache, index_cache):

    voronoi_area = [-1] * 600
    front_nodes = deque()
    voronoi = {}
    neutral_index = len(list_players)

    p_indexes = {}

    for player in list_players:
        p_pos = last_positions[player]
        p_indexes[player] = index_cache[p_pos[0]][p_pos[1]]
        voronoi_area[p_indexes[player]] = player
        front_nodes.append(p_pos)
        voronoi[player] = 1
    voronoi[neutral_index] = 0

    available_directions = [[0, -1], [0, 1], [-1, 0], [1, 0]]

    while front_nodes:
        cur = front_nodes.popleft()
        x, y = cur[0], cur[1]

        for off_x, off_y in available_directions:
            new_x = x + off_x
            new_y = y + off_y

            if 0 <= new_x < 30 and 0 <= new_y < 20 and area[index_cache[new_x][new_y]] and voronoi_area[index_cache[new_x][new_y]] == -1:

                min_distance = 1000
                min_player = -1

                for player in list_players:
                    distance = compute_path_array_cache(area, last_positions[player], p_indexes[player], (new_x,new_y), index_cache[new_x][new_y], manhattan_cache, index_cache)

                    if distance is not None:
                        if distance < min_distance:
                            min_distance = distance
                            min_player = player
                        elif distance == min_distance:
                            min_player = neutral_index

                voronoi_area[index_cache[new_x][new_y]] = min_player
                voronoi[min_player] += 1

                front_nodes.append((new_x, new_y))

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

    visited_area =  copy(area)

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

def heuristic_array(x0,y0,x1,y1):
    '''
    Heuristic for A*, here manhattan distance
    '''
    return abs(x0 - x1) + abs(y0 - y1)


def compute_path_array_heuristic(area, root, r_index, goal, g_index):
    '''
    implementation of A*
    '''

    pr_queue = []
    heappush(pr_queue, (0 + heuristic(root, goal), 0, root))

    visited_area = area[:]

    visited_area[r_index] = 0
    available_directions = [[0, -1], [0, 1], [-1, 0], [1, 0]]

    x_goal = goal[0]
    y_goal = goal[1]

    while len(pr_queue) > 0:
        _, cost, current = heappop(pr_queue)  # return the priority in the heap, cost, path and current element
        cur_index = current[1] * 30 + current[0]

        if cur_index == g_index: #Maybe change here to return the element and compute the path after ...
            return cost

        if visited_area[cur_index] == 0:
            visited_area[cur_index] = Configuration.WALL_CODE

            for off_x, off_y in available_directions:
                new_x = current[0] + off_x
                new_y = current[1] + off_y

                if 0 <= new_x < 30 and 0 <= new_y < 20:
                    neighbor = (new_x, new_y)
                    heappush(pr_queue, (cost + heuristic_array(new_x, new_y, x_goal, y_goal), cost + 1, neighbor))

    return None

def compute_path_array_cache(area, root, r_index, goal, g_index, cache_manhattan, cache_index):
    '''
    implementation of A*
    '''

    x_goal = goal[0]
    y_goal = goal[1]

    pr_queue = []
    heappush(pr_queue, (0 + cache_manhattan[root[0]][root[1]][x_goal][y_goal], 0, root))

    to_visit_area = area[:]

    to_visit_area[r_index] = True
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

def generate_manhattan_cache():
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

def generate_cache_index():
    index_cache = [None] * 30
    for x in range(30):
        for y in range(20):
            if index_cache[x] is None:
                index_cache[x] = [0] * 20

            index_cache[x][y] = x + 30 * y
    return index_cache