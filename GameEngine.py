import Configuration
from random import randint
from numpy import ones, zeros

class GameEngine:
    def __init__(self, nb_players):
        self.nb_players = nb_players
        self.cycles_positions = []
        self.area = ones((Configuration.MAX_X_GRID+1, Configuration.MAX_Y_GRID+1), dtype=bool)
        self.current_nb_players = nb_players
        self.players_game_over = []
        self.players = []

    def initialize(self, players):
        for i in range(self.nb_players):
            x,y = -1,-1
            while (x,y) in self.cycles_positions or (x == -1 and y == -1):
                x = randint(0, Configuration.MAX_X_GRID)
                y = randint(0, Configuration.MAX_Y_GRID)
            self.cycles_positions.append((x,y))
            self.players_game_over.append(False)

            players[i].initialize(self.cycles_positions[len(self.players)])
            self.players.append(players[i])

    def update(self):
        for i in range(self.nb_players):

            if not self.players_game_over[i]:
                is_game_over = False
                new_x = self.cycles_positions[i][0]
                new_y = self.cycles_positions[i][1]

                direction = self.players[i].compute_direction()
                print('direction = ' + direction, flush=True)
                if direction == 'LEFT': new_x -= 1
                elif direction == 'RIGHT': new_x += 1
                elif direction == 'UP': new_y -= 1
                elif direction == 'DOWN': new_y += 1
                else: #bad input => game over
                    is_game_over = True
                    print(str(i) + ' bad input', flush=True)

                if 0 <= new_x <= Configuration.MAX_X_GRID and 0 <= new_y <= Configuration.MAX_Y_GRID and self.area[new_x, new_y]:
                    self.cycles_positions[i] = (new_x, new_y)
                    self.area[new_x, new_y]= False
                else:
                    is_game_over = True
                    print(str(i) + ' out of bounds or hit wall', flush=True)

                if is_game_over:
                    self.current_nb_players -= 1
                    self.players_game_over[i] = True

    def is_game_playing(self):
        return self.current_nb_players > 1