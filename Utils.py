from collections import deque
from heapq import heappop, heappush
from time import clock
from math import inf

def compute_voronoi(area, last_positions, list_players, index_cache):

    #to_visit_area = area[:]
    #to_visit_area[index_cache[last_positions[0][0]][last_positions[0][1]]] = True
    #to_visit_area[index_cache[last_positions[1][0]][last_positions[1][1]]] = True

    front_nodes = deque()

    voronoi_area = [-1] * 600
    voronoi = {}
    for player in list_players:
        voronoi_area[index_cache[last_positions[player][0]][last_positions[player][1]]] = player
        front_nodes.append(last_positions[player])
        voronoi[player] = 0
    neutral_index = 5
    voronoi[neutral_index] = 0

    available_directions = [[0, -1], [0, 1], [-1, 0], [1, 0]]
    while front_nodes:
        cur = front_nodes.popleft()
        x, y = cur[0], cur[1]
        front_index = index_cache[x][y]

        neighbor_value = voronoi_area[front_index]

        if neighbor_value != neutral_index:
            for off_x, off_y in available_directions:
                new_x = x + off_x
                new_y = y + off_y

                if 0 <= new_x < 30 and 0 <= new_y < 20 and area[index_cache[new_x][new_y]] :
                    new_index = index_cache[new_x][new_y]

                    #to_visit_area[new_index] = False
                    next_value = voronoi_area[new_index]

                    if next_value == -1:
                        voronoi_area[new_index] = neighbor_value
                        voronoi[neighbor_value] += 1
                        front_nodes.append((new_x, new_y))
                    elif next_value != -1 and next_value != neighbor_value:
                        voronoi_area[new_index] = neutral_index
                        voronoi[neutral_index] += 1
                        voronoi[next_value] -= 1

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
                        new_chamber.space = 1
                        list_chambers.append(new_chamber)

                        chamber_area[new_index] = new_chamber
                        current_chamber.is_leaf = False
                        front_nodes.append((new_x, new_y))

                    elif chamber_area[new_index] is not None and chamber_area[new_index] != current_chamber and new_index != chamber_area[new_index].entrance and chamber_area[new_index] != origin_chamber:
                        #Meet a room that belongs to another room, merge the point to the new room
                        #Priority go to origin chamber that can not be merged
                        chamber_area[new_index].space -= 1
                        chamber_area[new_index] = current_chamber
                        current_chamber.space += 1

    # Step 2: Compute spaces between the different leaf chambers and root chamber
    # Step 3: Select best solution (more space => better solution)
    best_space = 0
    for chamber in list_chambers:
        if chamber.is_leaf:
            current_space = chamber.space
            parent = chamber.parent
            while parent != origin_chamber:
                current_space += parent.space
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

def scoring(initial_spaces, new_spaces, p_list_players, p_list_players_without_me, my_index, init_articulation_points, articulation_points, init_dist, new_dist):
    '''
    Define the score of the move depending of different factors
    '''

    is_ennemy_killed = False
    delta_ennemy_space = 0
    delta_distance = 0
    nb_separeted  = 0
    init_nb_separeted = 0
    max_ennemy = 0
    max_player = 0

    for player in p_list_players_without_me:
        #is_ennemy_killed = is_ennemy_killed or (initial_spaces[player] != 0 and new_spaces[player] == 0)

        #if init_dist_from_me[player] is not None and dist_from_me[player] is not None:
        #   delta_distance = (init_dist_from_me[player] - dist_from_me[player])/init_dist_from_me[player]
        #else:
        #    nb_separeted += 1
        #    delta_distance = 0

        #if init_dist[player] is None:
        #    init_nb_separeted +=1

        #if dist_from_me[player] is None:
        #    nb_separeted += 1

        if initial_spaces[player] != 0:
            delta_ennemy_space += (initial_spaces[player] - new_spaces[player])/initial_spaces[player]

        #if initial_spaces[player] > max_ennemy:
        #    max_ennemy = new_spaces[player]
        #    max_player = player

    delta_conflict_space = 0
    if initial_spaces[5] != 0:
        delta_conflict_space = (initial_spaces[5] - new_spaces[5])/initial_spaces[5]

    delta_my_space = 0
    if initial_spaces[my_index] != 0:
        delta_my_space = (new_spaces[my_index] - initial_spaces[my_index])/initial_spaces[my_index]

    nb_openings = 0
    #offsets = [[1, 0], [-1, 0], [0, 1], [0, -1]]
    #for off_x,off_y in offsets:
    #    new_x = new_pos[my_index][0] + off_x
    #    new_y = new_pos[my_index][1] + off_y
    #
    #    if 0 <= new_x < 30 and 0 <= new_y < 20 and area[index_cache[new_x][new_y]]:
    #        nb_openings += 1
    #nb_openings = nb_openings /4

    delta_articulation_points = 0
    if init_articulation_points != 0:
        delta_articulation_points = len(init_articulation_points) - len(articulation_points)

    bonus = 0
    if is_ennemy_killed:
        bonus += 3

    #Maximise my space
    #Minimize ennemy space
    #Bonus is great !

    if new_spaces[my_index] == 0:
        score = -inf
    else:
        score = delta_my_space*3 + delta_ennemy_space + delta_conflict_space *0.4+ bonus + nb_openings * (-0.0) + delta_distance*(0.007) + delta_articulation_points
        #2nd avec score = delta_my_space * 3.0 + delta_ennemy_space * 1.0 + delta_conflict_space * 0.4 + bonus
    return score
