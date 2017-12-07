import Configuration
from random import randint
from numpy import ones, zeros
from time import clock
from math import inf


class Statistics:
    def __init__(self, nb_players):
        self.nb_players = nb_players
        self.turn_of_death = inf
        self.players_killed_before = 0
        self.game_turns = 0

    def update(self, is_game_over, remaining_players):
        self.game_turns += 1

        if is_game_over and self.turn_of_death == inf:
            self.turn_of_death = self.game_turns
            remaining_players += 1 #Do not take into account player when counting deads

        if not is_game_over and remaining_players < self.nb_players:
            self.players_killed_before = self.nb_players - remaining_players

class GameEngine:

    def __init__(self, nb_players = 0):
        self.nb_players = nb_players
        self.cycles_positions = []
        self.initial_cycles_positions = []
        self.area = ones((Configuration.MAX_X_GRID+1, Configuration.MAX_Y_GRID+1), dtype=bool)
        self.current_nb_players = nb_players
        self.players_game_over = []
        self.players = []
        self.walls = {}
        self.statistics = {}

    def load_configuration(self, configuration, bots):
        '''
        Set the game depending of the configuration and the bots given in paramters
        :param configuration: GameConfiguration
        :param bots: list of Bots
        :return: None
        '''
        self.nb_players = configuration.nb_players
        self.current_nb_players = configuration.nb_players

        for i in range(configuration.nb_players):
            x = configuration.starting_positions[i][0]
            y = configuration.starting_positions[i][1]

            self.cycles_positions.append((x, y))
            self.initial_cycles_positions.append((x, y))
            self.players_game_over.append(False)
            self.players.append(bots[i])
            self.walls[i] = [(x, y)]

            self.statistics[i] = Statistics(configuration.nb_players)

    def initialize(self, players):
        for i in range(self.nb_players):
            x = Configuration.START_POSITIONS[i][0]
            y = Configuration.START_POSITIONS[i][1]

            self.cycles_positions.append((x,y))
            self.initial_cycles_positions.append((x,y))
            self.players_game_over.append(False)
            self.players.append(players[i])
            self.walls[i] = [(x,y)]

    def update(self):
        for i in range(self.nb_players):

            if not self.players_game_over[i]:
                is_game_over = False
                new_x = self.cycles_positions[i][0]
                new_y = self.cycles_positions[i][1]

                input = str(self.nb_players) + ' ' + str(i) + '\n'
                for j in range(self.nb_players):
                    if self.players_game_over[j]:
                        input += '-1 -1 -1 -1\n'
                    else:
                        input += str(self.initial_cycles_positions[j][0]) + ' ' + str(self.initial_cycles_positions[j][1]) + ' '
                        input += str(self.cycles_positions[j][0]) + ' ' + str(self.cycles_positions[j][1]) + '\n'

                direction = self.players[i].compute_direction(input)

                if direction == 'LEFT': new_x -= 1
                elif direction == 'RIGHT': new_x += 1
                elif direction == 'UP': new_y -= 1
                elif direction == 'DOWN': new_y += 1
                else: #bad input => game over
                    is_game_over = True

                if 0 <= new_x <= Configuration.MAX_X_GRID and 0 <= new_y <= Configuration.MAX_Y_GRID and self.area[new_x, new_y]:
                    self.cycles_positions[i] = (new_x, new_y)
                    self.walls[i].append((new_x, new_y))
                    self.area[new_x, new_y]= False
                else:
                    is_game_over = True

                if is_game_over:
                    self.current_nb_players -= 1
                    self.players_game_over[i] = True

                    #Free space
                    for wall in self.walls[i]:
                        self.area[wall[0], wall[1]] = True

            self.statistics[i].update(self.players_game_over[i], self.current_nb_players)

    def get_statistics(self):
        '''
        :return: statistics on the game
        '''

        return self.statistics

    def is_game_playing(self):
        return self.current_nb_players > 1