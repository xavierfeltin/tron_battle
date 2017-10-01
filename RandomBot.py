import Configuration
from Bot import Bot
from random import randint

class RandomBot(Bot):

    def __init__(self):
        Bot.__init__(self)
        self.position = (-1,-1)
        self.previous_position = (-1,-1)

    def compute_direction(self):
        direction = ''
        is_back = True
        while is_back:
            new_x = self.position[0]
            new_y = self.position[1]

            way = randint(0, 1)
            if way:
                offset = 1
            else:
                offset = -1

            orientation = randint(0, 1)
            if orientation:
                new_x = new_x + offset
                if new_x < 0:
                    new_x = 0
                elif new_x > Configuration.MAX_X_GRID:
                    new_x = Configuration.MAX_X_GRID
            else:
                new_y = new_y + offset
                if new_y < 0:
                    new_y = 0
                elif new_y > Configuration.MAX_Y_GRID:
                    new_y = Configuration.MAX_Y_GRID

            print('new position: ' + str(new_x) + ', ' + str(new_y), flush=True)

            if new_x - self.position[0] > 0: direction = 'RIGHT'
            elif new_x - self.position[0] < 0: direction = 'LEFT'
            elif new_y - self.position[1] > 0: direction = 'DOWN'
            elif new_y - self.position[1] < 0: direction = 'UP'
            else: direction = ''

            is_back = new_x == self.previous_position[0] and new_y == self.previous_position[1]

        self.previous_position = (self.position[0], self.position[1])
        self.position = (new_x, new_y)

        return direction