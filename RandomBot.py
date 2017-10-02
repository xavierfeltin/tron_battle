import Configuration
from Bot import Bot
from random import randint
from numpy import ones

class RandomBot(Bot):

    def __init__(self):
        Bot.__init__(self)
        self.position = (-1,-1)
        self.previous_position = (-1,-1)
        self.area = ones((Configuration.MAX_X_GRID+1, Configuration.MAX_Y_GRID+1), dtype=bool)

    def compute_direction(self, input):

        splitted = input.split('\n')

        nb_players, my_index = [int(i) for i in splitted[0].split()]
        for i in range(nb_players):
            # x0: starting X coordinate of lightcycle (or -1)
            # y0: starting Y coordinate of lightcycle (or -1)
            # x1: starting X coordinate of lightcycle (can be the same as X0 if you play before this player)
            # y1: starting Y coordinate of lightcycle (can be the same as Y0 if you play before this player)
            x0, y0, x1, y1 = [int(j) for j in splitted[i+1].split()]

            if i == my_index:
                self.position = (x1,y1)

            self.area[x0, y0] = False
            self.area[x1, y1] = False

        offsets = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        possibilities = []
        for offset in offsets:
            new_x = self.position[0] + offset[0]
            new_y = self.position[1] + offset[1]
            if 0 <= new_x <= Configuration.MAX_X_GRID and 0 <= new_y <= Configuration.MAX_Y_GRID and self.area[new_x, new_y]:
                if new_x - self.position[0] > 0: possibilities.append('RIGHT')
                elif new_x - self.position[0] < 0: possibilities.append('LEFT')
                elif new_y - self.position[1] > 0: possibilities.append('DOWN')
                elif new_y - self.position[1] < 0: possibilities.append('UP')

        if len(possibilities) > 0:
            rand = randint(0, len(possibilities)-1)
            return possibilities[rand]
        else:
            return ''