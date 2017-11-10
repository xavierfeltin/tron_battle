import Configuration
import numpy
from numpy import ones, zeros, copy, int32
from collections import deque

class Chamber:
    def __init__(self, p_entrance, p_parent = None):
        self.space = 0
        self.entrance = p_entrance
        self.positions = deque() #help in case of merge
        self.parent = p_parent

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

    '''
    https: // en.wikipedia.org / wiki / Biconnected_component
    GetArticulationPoints(i, d)
        visited[i] = true
        depth[i] = d
        low[i] = d
        childCount = 0
        isArticulation = false
        for each ni in adj[i]
            if not visited[ni]
                parent[ni] = i
                GetArticulationPoints(ni, d + 1)
                childCount = childCount + 1
                if low[ni] >= depth[i]
                    isArticulation = true
                low[i] = Min(low[i], low[ni])
            else if ni <> parent[i]
                low[i] = Min(low[i], depth[ni])


        if (parent[i] <> null and isArticulation) or (parent[i] == null and childCount > 1)
            Output
            i as articulation
            point
    '''

    '''
    Une implémentation du pseudo code de Wikipedia
    Je pense que c sert de depth dans cette implémentation
    https: // github.com / coreyabshire / tron / blob / master / tronutils.py
    V = set(); A = Adjacent(board, is_floor)
    L = {}; N = {}; c = [0]; P = {}; X = set()
    def f(v):
        V.add(v)
        c[0] += 1
        L[v] = N[v] = c[0]
        for w in A[v]:
            if w not in V:
                P[w] = v
                f(w)
                if v != root and L[w] >= N[v]:
                    X.add(v)
                L[v] = min(L[v], L[w])
            else:
                if v in P and P[v] != w:
                    L[v] = min(L[v], N[w])
    f(root)
    return X
    '''

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
    new_chamber = Chamber(previous_position)
    new_chamber.space = 1
    origin_chamber = new_chamber

    chamber_area[current_position] = new_chamber
    new_chamber.positions.append(current_position)
    front_nodes.append(current_position)

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
                        new_chamber = Chamber((new_x, new_y), current_chamber)
                        new_chamber.space = 1
                        new_chamber.positions.append((new_x, new_y))
                        list_chambers.append(new_chamber)

                        chamber_area[(new_x, new_y)] = new_chamber
                        front_nodes.append((new_x, new_y))
                    else:
                        if chamber_area[(new_x, new_y)] == current_chamber:
                            # Step 1-4: if neighbor associated with the current chamber (or entrance of the current chamber) => ignore it !
                            pass
                        elif chamber_area[(new_x, new_y)] != current_chamber and chamber_area[(new_x, new_y)] != 0 and current_chamber.entrance != (new_x, new_y):
                            #Step 1-5: if neighbor associated with a chamber different from the current chamber and not the entrance of the current chamber
                            # merge current chamber and neighbor chamber:
                                # identify the lowest common parent chamber
                                # merge the two chambers into the common parent
                            # do NOT add the neighbor to the front queue !

                            if current_chamber != origin_chamber: parent_1 = current_chamber.parent
                            else: parent_1 = current_chamber
                            if chamber_area[(new_x, new_y)] != origin_chamber: parent_2 = chamber_area[(new_x, new_y)].parent
                            else: parent_2 = chamber_area[(new_x, new_y)]

                            while parent_1 != parent_2:
                                if parent_1 != origin_chamber: parent_1 = parent_1.parent
                                if parent_2 != origin_chamber: parent_2 = parent_2.parent

                            if parent_1 != current_chamber:
                                for pos in current_chamber.positions:
                                    chamber_area[pos] = parent_1
                                    parent_1.positions.append(pos)
                                parent_1.space += current_chamber.space

                            if parent_1 != chamber_area[(new_x, new_y)]:
                                for pos in chamber_area[(new_x, new_y)].positions:
                                    chamber_area[pos] = parent_1
                                    parent_1.positions.append(pos)
                                parent_1.space += chamber_area[(new_x, new_y)].space

                            if current_chamber != origin_chamber: list_chambers.remove(current_chamber)
                            if chamber_area[(new_x, new_y)] != origin_chamber: list_chambers.remove(chamber_area[(new_x, new_y)])

    #Step 2: Compute spaces between the different leaf chambers and root chamber
    #Step 3: Select best solution (more space => better solution)
    best_space = 0
    for chamber in list_chambers:
        current_space = 0
        while chamber.parent != origin_chamber:
            current_space += chamber.space

        if best_space < current_space:
            best_space = current_space

    #No other chambers than the origin one
    best_space += origin_chamber.space

    return best_space