import numpy
from Bot import Bot
from collections import deque
from numpy import ones, copy


def compute_voronoi(cycles, list_players):
    voronoi_cells = {}
    for i in list_players:
        voronoi_cells[i] = 0

    for i in range(30):
        for j in range(20):
            distances = {}
            closest_cycle = -1
            is_limit = False

            for k in list_players:
                distances[k] = abs(i - cycles[k][0]) + abs(j - cycles[k][1])

                if closest_cycle == -1:
                    closest_cycle = k
                elif distances[k] < distances[closest_cycle]:
                    closest_cycle = k
                elif distances[k] == distances[closest_cycle]:
                    is_limit = True
                    break

            if not is_limit:
                voronoi_cells[closest_cycle] += 1

    return voronoi_cells

def process_availables_spaces(free_map, root, player_pos=None):
    nb_spaces = 0
    front_nodes = deque()

    area = numpy.copy(free_map)
    area[root] = False

    if player_pos is not None:
        area[player_pos] = False

    front_nodes.append(root)
    while len(front_nodes) > 0:
        nb_spaces += 1
        cur = front_nodes.popleft()

        available_directions = [[0, -1], [0, 1], [-1, 0], [1, 0]]

        for off_x, off_y in available_directions:
            new_x = cur[0] + off_x
            new_y = cur[1] + off_y

            if 0 <= new_x < 30 and 0 <= new_y < 20 and area[new_x, new_y]:
                area[new_x, new_y] = False
                front_nodes.append((new_x, new_y))

    return nb_spaces

class ExplicitBot(Bot):

    def __init__(self):
        Bot.__init__(self)
        self.position = (-1,-1)
        self.previous_position = (-1,-1)

        self.cur_cycles = {}
        self.wall_cycles = {}
        self.voronoi_cells = None
        self.my_index = 0
        self.nb_players = 0
        self.opposite_direction = (0, 0)
        self.free_map = ones((30, 20), dtype=bool)

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
            x0, y0, x1, y1 = [int(j) for j in splitted[i+1].split()]

            if x0 != -1:
                self.cur_cycles[i] = (x1, y1)

                if self.turn == 0:
                    self.wall_cycles[i] = [(x0, y0)]
                    self.free_map[x0, y0] = False

                self.wall_cycles[i].append((x1, y1))
                self.free_map[x1, y1] = False
            else:
                # If player has lost, remove his wall from the game
                if i in self.cur_cycles:
                    for case in self.wall_cycles[i]:
                        self.free_map[case] = True

                    del self.cur_cycles[i]
                    del self.wall_cycles[i]

                self.list_players.remove(i)
                self.list_players_without_me.remove(i)

        init_voronoi_cells = compute_voronoi(self.cur_cycles, self.list_players)

        available_directions = [(0, -1), (0, 1), (-1, 0), (1, 0)]

        # Remove directions causing to hit the arena wall
        if self.opposite_direction != (0, 0): available_directions.remove(self.opposite_direction)

        if self.cur_cycles[my_index][0] == 0:
            available_directions.remove((-1, 0))
        elif self.cur_cycles[my_index][0] == 29:
            available_directions.remove((1, 0))

        if self.cur_cycles[my_index][1] == 0:
            available_directions.remove((0, -1))
        elif self.cur_cycles[my_index][1] == 19:
            available_directions.remove((0, 1))

        new_cycles = {}
        for i in self.list_players:
            new_cycles[i] = self.cur_cycles[i]

        new_direction = (0, 0)
        max_nb_cases = 0

        min_ennemy_cases = 0
        for i in self.list_players_without_me:
            min_ennemy_cases += process_availables_spaces(self.free_map, self.cur_cycles[i], self.cur_cycles[my_index])

        max_voronoi = init_voronoi_cells[my_index]
        best_worst_voronoi = 0

        for direction in available_directions:
            new_x = self.cur_cycles[my_index][0] + direction[0]
            new_y = self.cur_cycles[my_index][1] + direction[1]

            if self.free_map[new_x, new_y]:
                new_cycles[my_index] = (new_x, new_y)
                nb_cases = process_availables_spaces(self.free_map, (new_x, new_y))

                if nb_cases > max_nb_cases:
                    new_direction = direction
                    max_nb_cases = nb_cases

                    new_voronoi_cells = compute_voronoi(new_cycles, self.list_players)
                    max_voronoi = new_voronoi_cells[my_index]

                    min_ennemy_cases = 0
                    for i in self.list_players_without_me:
                        min_ennemy_cases += process_availables_spaces(self.free_map, self.cur_cycles[i], (new_x, new_y))

                elif nb_cases == max_nb_cases:
                    ennemy_cases = 0
                    for i in self.list_players_without_me:
                        ennemy_cases += process_availables_spaces(self.free_map, self.cur_cycles[i], (new_x, new_y))

                    if ennemy_cases < min_ennemy_cases:
                        min_ennemy_cases = ennemy_cases
                        new_voronoi_cells = compute_voronoi(new_cycles, self.list_players)
                        max_voronoi = new_voronoi_cells[my_index]
                    elif ennemy_cases == min_ennemy_cases:
                        new_voronoi_cells = compute_voronoi(new_cycles, self.list_players)
                        if new_voronoi_cells[my_index] >= max_voronoi:
                            max_voronoi = new_voronoi_cells[my_index]
                            new_direction = direction

        # A single line with UP, DOWN, LEFT or RIGHT
        if new_direction == (0, 1):
            direction = 'DOWN'
        elif new_direction == (0, -1):
            direction = 'UP'
        elif new_direction == (1, 0):
            direction = 'RIGHT'
        elif new_direction == (-1, 0):
            direction = 'LEFT'
        else:
            direction = ''

        if new_direction != (0, 0):
            self.opposite_direction = (new_direction[0] * -1, new_direction[1] * -1)

        return direction
        self.turn += 1




