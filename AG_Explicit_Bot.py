import sys
from math import sqrt, log, inf
from random import randint
from time import clock
from ExplicitBot_optimized import OptimExplicitBot
from Utils import compute_path, detect_articulation_points, compute_tree_of_chambers, compute_voronoi, generate_index_cache, generate_manhattan_cache, scoring

class AGExplicitBot(OptimExplicitBot):
    def __init__(self, p_dms, p_des, p_dcs, p_kb, p_nbo, p_dd, p_dap):

        super().__init__()

        self.c_dms = p_dms  #Delta My Space
        self.c_des = p_des  # Delta Ennemies Space
        self.c_dcs  = p_dcs # Delta Conflict space
        self.c_kb  = p_kb   # Kill bonus
        self.c_nbo = p_nbo  # Number of openings after a move
        self.c_dd  = p_dd   # Delta distance with all the others players
        self.c_dap = p_dap  # Delta number of articulation points

    def scoring(self, initial_spaces, new_spaces, p_list_players, p_list_players_without_me, my_index,
                init_articulation_points, articulation_points, init_dist, new_dist, new_pos, init_dist_from_me, dist_from_me):
        '''
        Define the score of the move depending of different factors
        '''

        is_ennemy_killed = False
        delta_ennemy_space = 0
        delta_distance = 0
        nb_separeted = 0
        init_nb_separeted = 0
        max_ennemy = 0
        max_player = 0

        for player in p_list_players_without_me:
            # is_ennemy_killed = is_ennemy_killed or (initial_spaces[player] != 0 and new_spaces[player] == 0)

            if init_dist_from_me[player] is not None and dist_from_me[player] is not None:
               delta_distance = (init_dist_from_me[player] - dist_from_me[player])/init_dist_from_me[player]
            else:
                nb_separeted += 1
                delta_distance = 0

            if init_dist[player] is None:
                init_nb_separeted +=1

            if dist_from_me[player] is None:
                nb_separeted += 1

            if initial_spaces[player] != 0:
                delta_ennemy_space += (initial_spaces[player] - new_spaces[player]) / initial_spaces[player]

                # if initial_spaces[player] > max_ennemy:
                #    max_ennemy = new_spaces[player]
                #    max_player = player

        delta_conflict_space = 0
        if initial_spaces[5] != 0:
            delta_conflict_space = (initial_spaces[5] - new_spaces[5]) / initial_spaces[5]

        delta_my_space = 0
        if initial_spaces[my_index] != 0:
            delta_my_space = (new_spaces[my_index] - initial_spaces[my_index]) / initial_spaces[my_index]

        nb_openings = 0
        offsets = [[1, 0], [-1, 0], [0, 1], [0, -1]]
        for off_x,off_y in offsets:
            new_x = new_pos[my_index][0] + off_x
            new_y = new_pos[my_index][1] + off_y

            if 0 <= new_x < 30 and 0 <= new_y < 20 and self.area[self.index_cache[new_x][new_y]]:
                nb_openings += 1
        nb_openings = nb_openings/4

        delta_articulation_points = 0
        if init_articulation_points != 0:
            delta_articulation_points = len(init_articulation_points) - len(articulation_points)

        bonus = 0
        if is_ennemy_killed:
            bonus += 1

        # Maximise my space
        # Minimize ennemy space
        # Bonus is great !

        if new_spaces[my_index] == 0:
            score = -inf
        else:
            score = delta_my_space * self.c_dms + delta_ennemy_space * self.c_des + delta_conflict_space * self.c_dcs + bonus * self.c_kb + nb_openings * self.c_nbo + delta_distance * self.c_dd + delta_articulation_points * self.c_dap
            #ORIGINAL : score = delta_my_space * 3 + delta_ennemy_space + delta_conflict_space * 0.4 + bonus + nb_openings * (-0.0) + delta_distance * (0.007) + delta_articulation_points
        return score

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

        #Initial state evaluation
        init_voronoi_area, init_voronoi_spaces = compute_voronoi(self.area, self.current_move, self.list_players, self.index_cache)

        init_dist_from_me = {}
        r_index = self.index_cache[self.current_move[my_index][0]][self.current_move[my_index][1]]
        for player in self.list_players_without_me:
            g_index = self.index_cache[self.current_move[player][0]][self.current_move[player][1]]
            init_dist_from_me[player] = compute_path(self.area, self.current_move[my_index], r_index, self.current_move[player],
                                                     g_index, self.manhattan_cache, self.index_cache)

        init_previous_pos = {}
        for player in self.list_players:
            if len(self.wall_cycles[player]) < 2:
                init_previous_pos[player] = (-1,-1)
            else:
                init_previous_pos[player] = self.wall_cycles[player][-2]

        init_articulation_points = {}  # Another solution is to detect which players are separated from each other with A*..
        init_availables_spaces = {}
        for player in self.list_players:
            init_articulation_points[player] = detect_articulation_points(self.area, self.current_move[player],
                                                                     self.index_cache[self.current_move[player][0]][
                                                                         self.current_move[player][1]], self.index_cache)

            if len(init_articulation_points[player]) == 0:
                init_availables_spaces[player] = init_voronoi_spaces[player]
            else:
                init_availables_spaces[player] = compute_tree_of_chambers(self.area, init_voronoi_area,
                                                                             init_articulation_points[player], self.current_move[player],
                                                                             init_previous_pos, self.index_cache, player)
        init_availables_spaces[5] = init_voronoi_spaces[5]

        init_ennemies_space = 0
        for player in self.list_players_without_me:
            init_ennemies_space += init_availables_spaces[player]
        init_ennemies_space += init_voronoi_spaces[5]

        #Evaluation of the different possibles moves
        offsets = [[1, 0], [-1, 0], [0, 1], [0, -1]]
        cur_pos = self.current_move[my_index]

        new_pos = {}
        previous_pos = {}
        for player in self.list_players:
            new_pos[player] = self.current_move[player]
            previous_pos[player] = self.wall_cycles[player][-1]

        max_evaluation = -inf
        max_new_direction = (0,0)

        for off_x, off_y in offsets:
            new_x = cur_pos[0] + off_x
            new_y = cur_pos[1] + off_y

            if 0 <= new_x < 30 and 0 <= new_y < 20 and self.area[self.index_cache[new_x][new_y]]:
                previous_pos[my_index] = new_pos[my_index]
                new_pos[my_index] = (new_x, new_y)

                voronoi_area, voronoi_spaces = compute_voronoi(self.area, new_pos, self.list_players, self.index_cache)

                articulation_points = {} #Another solution is to detect which players are separated from each other with A*..
                availables_spaces = {}
                dist_from_me = {}
                r_index = self.index_cache[new_pos[my_index][0]][new_pos[my_index][1]]
                for player in self.list_players:
                    g_index = self.index_cache[new_pos[player][0]][new_pos[player][1]]
                    dist_from_me[player] = compute_path(self.area, new_pos[my_index], r_index, new_pos[player], g_index,
                                                        self.manhattan_cache, self.index_cache)

                    articulation_points[player] = detect_articulation_points(self.area, new_pos[player], self.index_cache[new_pos[player][0]][new_pos[player][1]], self.index_cache)

                    if len(articulation_points[player]) == 0:
                        availables_spaces[player] = voronoi_spaces[player]
                    else:
                        availables_spaces[player] = compute_tree_of_chambers(self.area, voronoi_area, articulation_points[player], new_pos[player], previous_pos, self.index_cache, player)
                availables_spaces[5] = voronoi_spaces[5]

                ennemies_space = 0
                for player in self.list_players_without_me:
                    ennemies_space += availables_spaces[player]
                ennemies_space += voronoi_spaces[5]

                evaluation = self.scoring(init_availables_spaces, availables_spaces, self.list_players, self.list_players_without_me, my_index,
                                     init_articulation_points, articulation_points, init_dist_from_me, dist_from_me, new_pos, init_dist_from_me, dist_from_me)

                #print('player' + str(my_index) + ' cur:' + str((cur_pos[0],cur_pos[1])) + ' =>' + str((new_x,new_y)) + ': init:' + str(init_availables_spaces[my_index]) + '/' + str(init_ennemies_space) + ', new: ' + str(availables_spaces[my_index]) + '/' + str(ennemies_space) + ', score: ' + str(evaluation ) + ', ' + str(voronoi_spaces), flush=True)

                if evaluation > max_evaluation:
                    max_evaluation = evaluation
                    max_new_direction = (new_x, new_y)

        #print('player' + str(my_index) + ' cur:' + str((cur_pos[0], cur_pos[1])) + ' =>' + str((max_new_direction[0], max_new_direction[1])), flush=True)

        if max_new_direction != (0, 0):
            self.opposite_direction = (max_new_direction[0] * -1, max_new_direction[1] * -1)

        direction = ''
        if max_new_direction[0] - cur_pos[0] > 0: direction = 'RIGHT'
        elif max_new_direction[0] - cur_pos[0] < 0: direction = 'LEFT'
        elif max_new_direction[1] - cur_pos[1] > 0: direction = 'DOWN'
        elif max_new_direction[1] - cur_pos[1] < 0: direction = 'UP'

        self.turn += 1

        return direction