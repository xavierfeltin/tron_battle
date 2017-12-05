from random import randint, uniform
from math import inf
from operator import attrgetter
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
        game.my_position = randint(0, game.nb_players)

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
    def __init___(self, p_coefficients):
        self.coefficients = p_coefficients
        self.score = 0
        self.statistics = {}

    def add_statistics(self, index_game, statistic):
        self.statistics[index_game] = statistic

    def score(self):
        for stat in self.statistics:
            if stat.turn_of_death == inf: #player wins
                self.score += 1
            else:
                self.score += stat.turn_of_death / stat.game_turns #more turns before death is better
        self.score = self.score / len(self.statistics)


class Solver:
    def __init__(self, game, bots):
        self.game = game
        self.bots = bots
        self.engine = GameEngine()

    def solve(self):
        self.engine.load_configuration(self.game, self.bots)
        game_statistics = self.__game_loop(self.engine)
        return game_statistics

    def __game_loop(self):
        while self.engine.is_game_playing():
            self.engine.update()

        return self.engine.get_statistics()

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

        while self.index_generation <= self.min_evaluations and abs(self.best_score - self.previous_score) > 0.1:
            self.__solve()

            self.previous_score = self.best_score
            self.best_score = self.get_best_solution().score

        return self.get_best_solution().coefficients

    def __solve(self):
        '''
        Evaluate the current generation
        :return: None
        '''

        games = GameConfiguration.create_games(self.nb_games)

        for individual in self.population:
            for game in games:
                bots = []
                for i in range(game.nb_players):
                    if i == game.my_position:
                        p_ag_parameters = [3, 1, 0.4, 3, 0, 0.007, 1]
                        bots.append(AGExplicitBot(*p_ag_parameters))
                    else:
                        bots.append(OptimExplicitBot())

                solver = Solver(game, bots)
                game_statistics = solver.solve()

                individual.add_statistics(game_statistics, self.index_generation)

                print('config: ' + str(game.nb_players) + ', ' + str(game.my_position) + ', ' + str(game.starting_positions))
                print('stats: ' + str(game_statistics.turn_of_death) + ', ' + str(game_statistics.players_killed_before) + ', ' + str(game_statistics.game_turns))

    def __evaluate(self):
        '''
        Evaluate all the population based on the games previously played
        Sort the population by score
        :return: None
        '''
        for individual in self.population:
            individual.score()

        self.population.sort(key=attrgetter('score'), reverse=True)

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

                if self.solutions[index_opponent].score > winner.score:
                    index_winner = index_opponent
                    winner = self.solutions[index_opponent]

            if winner.score> maximum_score:
                maximum_score = winner.score

            self.parents.append(index_winner)
        return maximum_score

    def crossing_mutation_single(self):
        parent_1 = self.solutions[self.parents[randint(0, self.nb_tournament)]]
        parent_2 = self.solutions[self.parents[randint(0, self.nb_tournament)]]

        #TODO: to adapt to Tron
        child = Individual()
        p1_moves1 = parent_1.moves1
        p2_moves1 = parent_2.moves1
        p1_moves2 = parent_1.moves2
        p2_moves2 = parent_2.moves2

        for j in range(NB_MOVES):
            child.moves1.append(Move.cross(p1_moves1[j], p2_moves1[j], cho))
            child.moves2.append(Move.cross(p1_moves2[j], p2_moves2[j], gall))
        child.validate()

        return child

    def build_generation_proba(self):

        maximum_parent = self.tournament()
        crossing_probability = self.compute_probability_crossing(maximum_parent)
        new_average = 0.0
        nb_children = 0

        self.solutions.append(self.solutions[0].clone())

        for i in range(self.population_size):

            solution = self.solutions[i]
            if self.apocalypse < self.apocalypse_threshold:
                mutation_probability = self.compute_probability_mutation(solution.score)

                if uniform(0.0, 1.0) <= mutation_probability:
                    solution.mutate(mutation_probability)
                    solution.score()
                    new_average += solution.score
            else:
                solution.mutate(self.apocalypse_mutation_factor, True)
                solution.score()
                new_average += solution.score

            child = None
            if uniform(0.0, 1.0) <= crossing_probability:  # and nb_children <= MAX_NB_CHILDREN:
                child = self.crossing_mutation_single()
                child.score()
                self.solutions.append(child)
                new_average += child.score
                nb_children += 1

        self.population.sort(key=attrgetter('score'), reverse=True)
        self.population = self.population[0:self.population_size-1]

        self.average = new_average / self.population_size

        if self.population[0].score > self.maximum:
            self.maximum = self.population[0].score
            self.apocalypse = 0
        else:
            self.apocalypse += 1

    def update_avg_max(self, score):

        self.average += score / self.population_size
        if score > self.maximum:
            self.maximum = score

    def compute_probability_crossing(self, maximum_parent):
        if maximum_parent >= self.average:
            return max(self.K1 * ((self.maximum - maximum_parent) / (self.maximum - self.average)), self.min_proba_cross)
        else:
            return self.K3

    def compute_probability_mutation(self, score):
        if score >= self.average:
            return max(self.K2 * ((self.maximum - score) / (self.maximum - self.average)), self.min_proba_mutation)
        else:
            return self.K4

    def get_best_solution(self):
        return self.solutions[0]

    def generate_population(self, is_first_generation):
        '''
        Generate the population of solutions
        The populations is sorted with the best solution first
        @param is_first_generation: True if the generation is the really first one
        @return: None
        '''

        self.maximum = -inf
        self.average = 0.0

        #TODO: adapt to Tron

        if is_first_generation:
            reference_solution = Individual()
            reference_solution.generate_deterministic_solution(True)
            reference_solution.score()
            self.population.append(reference_solution)
            self.update_avg_max(reference_solution.score)

            for i in range(self.population_size - 1):
                solution = Individual()
                solution.generate_solution_from_reference(reference_solution, True)
                solution.score()
                self.population.append(solution)
                self.update_avg_max(solution.score)
        else:
            self.population[0].generate_deterministic_solution(False)
            self.population[0].score()
            self.update_avg_max(self.population[0].score)

            for i in range(1, self.population_size - 1):
                self.population[i + 1].generate_solution_from_reference(self.population[0], False)
                self.population[i + 1].score()
                self.update_avg_max(self.population[i + 1].score)

        self.population.sort(key=attrgetter('score'), reverse=True)