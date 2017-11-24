import sys
from math import sqrt, log, inf
from random import randint
from time import clock
from Bot import Bot
from Utils import compute_path, detect_articulation_points, compute_tree_of_chambers, compute_voronoi, generate_index_cache, generate_manhattan_cache

class ExplicitBot(Bot):
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

                    if i < my_index:  # players playing before me have already made a move
                        self.previous_move[i] = (x0, y0)
                        self.current_move[i] = (x1, y1)
                        self.wall_cycles[i] = [(x1, y1)]
                        self.area[self.index_cache[x1][y1]] = False
                else:
                    self.previous_move[i] = self.current_move[i]
                    self.current_move[i] = (x1,y1)
                    self.wall_cycles[i].append((x1, y1))
                    self.area[self.index_cache[x1][y1]] = False
            else:
                # If player has lost, remove his wall from the game
                if i in self.current_move:
                    for case in self.wall_cycles[i]:
                        self.area[self.index_cache[case[0]][case[1]]] = True

                    del self.current_move[i]
                    del self.wall_cycles[i]

                    self.list_players.remove(i)
                    self.list_players_without_me.remove(i)

        offsets = [[1, 0], [-1, 0], [0, 1], [0, -1]]
        cur_pos = self.current_move[my_index]

        new_pos = {}
        previous_pos = {}
        for player in self.list_players:
            new_pos[player] = self.current_move[player]
            previous_pos[player] = self.wall_cycles[player][-1]

        max_space = 0
        max_new_direction = (0,0)
        min_ennemies_space = inf

        for off_x, off_y in offsets:
            new_x = cur_pos[0] + off_x
            new_y = cur_pos[1] + off_y

            if 0 <= new_x < 30 and 0 <= new_y < 20 and self.area[self.index_cache[new_x][new_y]]:
                previous_pos[my_index] = new_pos[my_index]
                new_pos[my_index] = (new_x, new_y)

                voronoi_area, voronoi_spaces = compute_voronoi(self.area, new_pos, self.list_players, self.index_cache)

                articulation_points = {} #Another solution is to detect which players are separated from each other with A*..
                availables_spaces = {}
                for player in self.list_players:
                    articulation_points[player] = detect_articulation_points(self.area, new_pos[player], self.index_cache[new_pos[player][0]][new_pos[player][1]], self.index_cache)

                    if len(articulation_points[player]) == 0:
                        availables_spaces[player] = voronoi_spaces[player]
                    else:
                        availables_spaces[player] = compute_tree_of_chambers(self.area, voronoi_area, articulation_points[player], new_pos[player], previous_pos, self.index_cache, my_index)

                ennemies_space = 0
                max_ennemi_space = 0
                for player in self.list_players_without_me:
                    if availables_spaces[player] > max_ennemi_space :
                        max_ennemi_space = availables_spaces[player]
                    #ennemies_space += availables_spaces[player]
                ennemies_space = max_ennemi_space
                #ennemies_space += voronoi_spaces[5]

                #print('player-' + str(my_index) + ' cur:' + str((cur_pos[0],cur_pos[1])) + ', new:' + str((new_x,new_y)) + ': ' + str(availables_spaces[my_index]) + '/' + str(ennemies_space) + ', ' + str(voronoi_spaces), flush=True)

                #First maximise my own space
                if availables_spaces[my_index] > max_space:
                    max_space = availables_spaces[my_index]
                    max_new_direction = (new_x, new_y)
                    min_ennemies_space = ennemies_space

                #Second minimize my ennemies space
                elif availables_spaces[my_index] == max_space:
                    if ennemies_space < min_ennemies_space:
                        min_ennemies_space = ennemies_space
                        max_new_direction = (new_x, new_y)

        if max_new_direction != (0, 0):
            self.opposite_direction = (max_new_direction[0] * -1, max_new_direction[1] * -1)

        direction = ''
        if max_new_direction[0] - cur_pos[0] > 0: direction = 'RIGHT'
        elif max_new_direction[0] - cur_pos[0] < 0: direction = 'LEFT'
        elif max_new_direction[1] - cur_pos[1] > 0: direction = 'DOWN'
        elif max_new_direction[1] - cur_pos[1] < 0: direction = 'UP'

        self.turn += 1

        return direction