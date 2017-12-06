from random import randint, uniform
from math import inf
from time import clock
from operator import attrgetter
from threading import Thread

from AG_Explicit_Bot import AGExplicitBot
from ExplicitBot_optimized import OptimExplicitBot
from GameEngine import GameEngine

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
        game.my_position = randint(0, game.nb_players-1)

        positions = []
        for i in range(game.nb_players):
            x = randint(0, 29)
            y = randint(0, 19)

            if (x,y) not in positions:
                game.starting_positions[i] = (x,y)

        return game

    @staticmethod
    def create_games(p_nb_games):
        games = []
        for i in range(0, p_nb_games):
            games.append(GameConfiguration.create_game())
        return games


class Individual:
    def __init__(self, p_coefficients):
        self.coefficients = p_coefficients
        self.score = 0
        self.statistics = {}

    def add_statistics(self, index_game, statistic):
        self.statistics[index_game] = statistic

    def evaluate(self):
        for index, stat in self.statistics.items():
            if stat.turn_of_death == inf: #player wins
                self.score += 1
            else:
                self.score += stat.turn_of_death / stat.game_turns #more turns before death is better
        self.score = self.score / len(self.statistics)

    def clone(self):
        clone = Individual(self.coefficients[:])
        return clone

    def mutate(self, amplitude, is_apocalypse=False):

        minimums = []
        maximums = []

        for i in range(len(self.coefficients)):
            minimums.append(self.coefficients[i] - 10 * amplitude)
            maximums.append(self.coefficients[i] + 10 * amplitude)

            self.coefficients[i] = uniform(minimums[i], maximums[i])
            self.coefficients[i] = min(self.coefficients[i], 5)
            self.coefficients[i] = max(self.coefficients[i], -5)

    @staticmethod
    def get_actual_coefficients():
        '''
        Actual solution in codingame
        '''
        return [3, 1, 0.4, 3, 0, 0.007, 1]

    @staticmethod
    def generate_individual_from_reference(reference_individual):
        '''
        Create a new solution from an existing solution by making mutation on it
        @is_first_generation: True generate all moves, False delete first move and generate last one
        @return None
        '''
        amplitude = uniform(0,1)
        individual = reference_individual.clone()
        individual.mutate(amplitude)
        return individual

    @staticmethod
    def cross(individual1, individual2):

        crossed_coefficients = []
        for i in range(len(individual1.coefficients)):
            crossed_coefficients.append((individual1.coefficients[i] + individual2.coefficients[i]) / 2)

        child = Individual(crossed_coefficients)
        return child

class Solver:
    def __init__(self, game, bots):
        self.game = game
        self.bots = bots
        self.engine = GameEngine()

    def solve(self):
        self.engine.load_configuration(self.game, self.bots)
        game_statistics = self.__game_loop()
        return game_statistics

    def __game_loop(self):
        while self.engine.is_game_playing():
            self.engine.update()

        return self.engine.get_statistics()

class ComputeGame(Thread):
    def __init__(self, games, game_bots, individual, index_generation):
        Thread.__init__(self)
        self.games = games
        self.game_bots = game_bots
        self.individual = individual
        self.index = index_generation

    def run(self):
        for index_game, game in enumerate(self.games):
            solver = Solver(game, self.game_bots[index_game])
            game_statistics = solver.solve()
            self.individual.add_statistics( self.index, game_statistics[solver.game.my_position])

class AG:
    def __init__(self, population_size = 50, nb_games = 100, min_evaluations = 100, nb_tournament = 30, nb_tournament_contestants= 2, apocalypse_threshold = inf, apocalypse_mutation_factor = 0.5):
        self.population_size = population_size
        self.nb_games = nb_games
        self.min_evaluations = min_evaluations
        self.index_generation = 0
        self.nb_tournament = nb_tournament
        self.nb_tournament_contestants = nb_tournament_contestants
        self.apocalypse_threshold = apocalypse_threshold
        self.apocalypse_mutation_factor = apocalypse_mutation_factor

        # SPECIFIC TO ADAPTATIVE GENETIC ALGORITHM
        self.K1 = 1.0  # ponderation for crossing probability, 1.0 from publication
        self.K3 = 1.0  # ponderation for crossing probability, 1.0 from publication
        self.K2 = 0.5  # ponderation for mutation probability, 0.5 from publication
        self.K4 = 0.5  # ponderation for mutation probability, 0.5 from publication
        self.min_proba_mutation = 0.005  # minimum proba of mutation even on best solution 0.005 from publication
        self.min_proba_cross = 0.1  #own experiment

        self.best_score = -inf
        self.previous_score = -inf

        self.reference_individual = None
        self.population = [] #ordered by score
        self.parents = []
        self.children = []  #used only for the preselection approach

        self.average = 0.0
        self.maximum = 0.0
        self.apocalypse = 0

    def run(self):
        '''
        Run the algorithm
        :return: the best estimated coefficients
        '''

        while self.index_generation <= self.min_evaluations or abs(self.best_score - self.previous_score) > 0.01:
            print('Generation ' + str(self.index_generation))

            if self.index_generation == 0:
                self.generate_initial_population()
            else:
                self.build_generation_proba()

            self.__solve()
            self.__evaluate()

            self.previous_score = self.best_score
            self.best_score = self.get_best_solution().score

            self.index_generation += 1
            print('best generation score: ' + str(self.best_score) + ', reference score: ' + str(self.reference_individual.score), flush=True)

        return self.get_best_solution().coefficients

    def __solve(self):
        '''
        Evaluate the current generation
        :return: None
        '''

        games = GameConfiguration.create_games(self.nb_games)

        for index_ind, individual in enumerate(self.population):
            print('Evaluating indiv ' + str(index_ind), flush=True)
            threads = []

            game_bots = {}
            for index_game, game in enumerate(games):
                bots = []
                for i in range(game.nb_players):
                    if i == game.my_position:
                        p_ag_parameters = individual.coefficients
                        bots.append(AGExplicitBot(*p_ag_parameters))
                    else:
                        bots.append(OptimExplicitBot())
                game_bots[index_game] = bots

            threads.append(ComputeGame(games, game_bots, individual, index_game))

        for thread in threads:
            thread.start()

        for thread in threads:
            thread.join()

        print('    Evaluating Reference', flush=True)
        self.reference_individual = Individual(Individual.get_actual_coefficients())
        for index_game, game in enumerate(games):
            bots = []
            for i in range(game.nb_players):
                if i == game.my_position:
                    p_ag_parameters = self.reference_individual.coefficients
                    bots.append(AGExplicitBot(*p_ag_parameters))
                else:
                    bots.append(OptimExplicitBot())

            solver = Solver(game, bots)
            game_statistics = solver.solve()

            self.reference_individual.add_statistics(index_game, game_statistics[solver.game.my_position])

    def __evaluate(self):
        '''
        Evaluate all the population based on the games previously played
        Sort the population by score
        :return: None
        '''
        self.maximum = -inf
        self.average = 0.0

        for individual in self.population:
            individual.evaluate()
            self.update_avg_max(individual.score)

        self.population.sort(key=attrgetter('score'), reverse=True)
        self.population = self.population[0:self.population_size]

        self.reference_individual.evaluate()


    def __tournament(self):
        '''
        Select the parents for crossing with the tournament algorithm
        @return: highest score of selected parents
        '''
        self.parents.clear()
        self.parents.append(0)

        maximum_score = -inf

        for i in range(self.nb_tournament):
            index_winner = randint(0, self.population_size-1)
            winner = self.population[index_winner]
            for j in range(self.nb_tournament_contestants):
                index_opponent = randint(0, self.population_size - 1)

                if self.population[index_opponent].score > winner.score:
                    index_winner = index_opponent
                    winner = self.population[index_opponent]

            if winner.score> maximum_score:
                maximum_score = winner.score

            self.parents.append(index_winner)
        return maximum_score

    def crossing_mutation_single(self):
        parent_1 = self.population[self.parents[randint(0, self.nb_tournament)]]
        parent_2 = self.population[self.parents[randint(0, self.nb_tournament)]]

        return Individual.cross(parent_1, parent_2)

    def build_generation_proba(self):

        maximum_parent = self.__tournament()
        crossing_probability = self.compute_probability_crossing(maximum_parent)
        new_average = 0.0
        nb_children = 0

        self.population.append(self.get_best_solution().clone())

        for i in range(self.population_size):

            individual = self.population[i]
            if self.apocalypse < self.apocalypse_threshold:
                mutation_probability = self.compute_probability_mutation(individual.score)

                if uniform(0.0, 1.0) <= mutation_probability:
                    individual.mutate(mutation_probability)
            else:
                individual.mutate(self.apocalypse_mutation_factor, True)

            if uniform(0.0, 1.0) <= crossing_probability:
                child = self.crossing_mutation_single()
                self.population.append(child)
                nb_children += 1

        '''
        if self.population[0].score > self.maximum:
            self.maximum = self.population[0].score
            self.apocalypse = 0
        else:
            self.apocalypse += 1
        '''

    def update_avg_max(self, score):

        self.average += score / self.population_size
        if score > self.maximum:
            self.maximum = score

    def compute_probability_crossing(self, maximum_parent):
        if maximum_parent > self.average:
            return max(self.K1 * ((self.maximum - maximum_parent) / (self.maximum - self.average)), self.min_proba_cross)
        else:
            return self.K3

    def compute_probability_mutation(self, score):
        if score > self.average:
            return max(self.K2 * ((self.maximum - score) / (self.maximum - self.average)), self.min_proba_mutation)
        else:
            return self.K4

    def get_best_solution(self):
        return self.population[0]

    def generate_initial_population(self):
        '''
        Generate the population of solutions
        The populations is sorted with the best solution first
        @param is_first_generation: True if the generation is the really first one
        @return: None
        '''
        ref_individual = Individual(Individual.get_actual_coefficients())

        for i in range(self.population_size):
            self.population.append(Individual.generate_individual_from_reference(ref_individual))