from random import randint

MAX_X_GRID = 29
MAX_Y_GRID = 19

class GameEngine:
    def __init__(self, nb_players):
        self.nb_players = nb_players
        self.cycles_positions = []
        self.nb_turns = 2000

    def initialize(self):
        for i in range(self.nb_players):
            x,y = -1,-1
            while (x,y) in self.cycles_positions or (x == -1 and y == -1):
                x = randint(0, MAX_X_GRID)
                y = randint(0, MAX_Y_GRID)
            self.cycles_positions.append((x,y))

    def update(self):
        for i in range(self.nb_players):
            new_x = self.cycles_positions[i][0]
            new_y = self.cycles_positions[i][1]

            way = randint(0,1)
            if way: offset = 1
            else: offset = -1

            direction = randint(0, 1)
            if direction:
                new_x = new_x + offset
                if new_x < 0: new_x = 0
                elif new_x > MAX_X_GRID: new_x = MAX_X_GRID
            else:
                new_y = new_y  + offset
                if new_y < 0: new_y = 0
                elif new_y > MAX_Y_GRID: new_y = MAX_Y_GRID

            self.cycles_positions[i] = (new_x, new_y)

    def is_game_playing(self):
        self.nb_turns -= 1
        return self.nb_turns > 0