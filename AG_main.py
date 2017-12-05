import threading
import gc
from random import randint
from GameFrame import GameFrame
from GameEngine import GameEngine
from ExplicitBot_optimized import OptimExplicitBot
from AG_Explicit_Bot import AGExplicitBot
from time import sleep

MAX_NB_PLAYERS = 4

class GameConfiguration:
    def __init__(self):
        self.nb_players = 0
        self.my_position = 0
        self.starting_positions = {}

    @staticmethod
    def create_game():
        game = GameConfiguration()
        game.nb_players = randint(2, MAX_NB_PLAYERS)
        game.my_position = randint(0, game.nb_players)

        positions = []
        for i in range(game.nb_players):
            x = randint(0, 29)
            y = randint(0, 19)

            if (x,y) not in positions:
                game.starting_positions[i] = (x,y)

        return game

def game_generator(p_nb_games):
    games = []
    for i in range(0, p_nb_games):
        games.append(GameConfiguration.create_game())
    return games

def game_loop(engine):
    gc.disable()
    while engine.is_game_playing():
        engine.update()
    gc.enable()

    return engine.get_statistics()

if __name__ == "__main__":

    population = [0] * 1
    games = game_generator(3)

    for individual in population:
        for game in games:
            bots = []
            for i in range(game.nb_players):
                if i == game.my_position:
                    p_ag_parameters = [3, 1, 0.4, 3, 0, 0.007, 1]
                    bots.append(AGExplicitBot(*p_ag_parameters))
                else:
                    bots.append(OptimExplicitBot())

            engine = GameEngine()
            engine.load_configuration(game, bots)
            game_statistics = game_loop(engine)

            #individual.evaluate(game_statistics)

            print('config: ' + str(game.nb_players) + ', ' + str(game.my_position) + ', ' + str(game.starting_positions))
            print('stats: ' + str(game_statistics.turn_of_death) + ', ' + str(game_statistics.players_killed_before) + ', ' + str(game_statistics.game_turns))

    #Selection and so on...

