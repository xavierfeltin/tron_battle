import numpy
from Bot import Bot
from collections import deque
from numpy import ones, copy

class MCTSBot():
    def __init__(self):
        self.cur_cycles = {}
        self.wall_cycles = {}
        self.area = ones((30, 20), dtype=bool)
        self.list_players = []
        self.list_players_without_me = []
        self.turn = 0

    def compute_mcts(area, list_players, my_index):
        return ''

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
                    self.area[x0, y0] = False

                self.wall_cycles[i].append((x1, y1))
                self.area[x1, y1] = False
            else:
                # If player has lost, remove his wall from the game
                if i in self.cur_cycles:
                    for case in self.wall_cycles[i]:
                        self.area[case] = True

                    del self.cur_cycles[i]
                    del self.wall_cycles[i]

                    self.list_players.remove(i)
                    self.list_players_without_me.remove(i)

        direction = compute_mcts(self.area, self.list_players, my_index)

        return direction