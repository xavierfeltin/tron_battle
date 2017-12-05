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

    def clone(self):
        clone = Individual(self.coefficients[:])
        return clone

    @staticmethod
    def get_actual_coefficients():
        '''
        Actual solution in codingame
        '''
        return [0.3, 0.1, 0.04, 0.3, 0, 0.0007, 0.1]

    def generate_individual_from_reference(self, reference_individual):
        '''
        Create a new solution from an existing solution by making mutation on it
        @is_first_generation: True generate all moves, False delete first move and generate last one
        @return None
        '''

        if is_first_generation:
            for i in range(NB_MOVES):
                move = reference_solution.moves1[i].clone()
                move.mutate(uniform(COEFFICIENT_MIN_MUTATION_FROM_REF, COEFFICIENT_MAX_MUTATION_FROM_REF), cho)
                self.moves1.append(move)

                move = reference_solution.moves2[i].clone()
                move.mutate(uniform(COEFFICIENT_MIN_MUTATION_FROM_REF, COEFFICIENT_MAX_MUTATION_FROM_REF), gall)
                self.moves2.append(move)

            self.validate()
        else:
            move = reference_solution.moves1[NB_MOVES - 1].clone()
            move.mutate(uniform(COEFFICIENT_MIN_MUTATION_FROM_REF, COEFFICIENT_MAX_MUTATION_FROM_REF), cho)
            self.moves1.popleft()
            self.moves1.append(move)

            move = reference_solution.moves2[NB_MOVES - 1].clone()
            move.mutate(uniform(COEFFICIENT_MIN_MUTATION_FROM_REF, COEFFICIENT_MAX_MUTATION_FROM_REF), gall)
            self.moves2.popleft()
            self.moves2.append(move)

        self.validate()

    def mutate(self, amplitude, is_apocalypse=False):

        if race_turn > 2 and not is_apocalypse:
            for i in reversed(range(NB_MOVES_TO_MUTATE)):
                self.moves1[NB_MOVES - 1 - i].mutate(amplitude, cho)
                self.moves2[NB_MOVES - 1 - i].mutate(amplitude, gall)
        else:
            for i in range(NB_MOVES):
                self.moves1[i].mutate(amplitude, cho)
                self.moves2[i].mutate(amplitude, gall)

        self.validate()

    def mutate(self, amplitude, pod):
        ramin = self.angle - 36.0 * amplitude
        ramax = self.angle + 36.0 * amplitude

        if ramin < -18.0:
            ramin = -18.0

        if ramax > 18.0:
            ramax = 18.0

        self.angle = uniform(ramin, ramax)

        self.shield = pod.shield_ready and randint(0, 100) < SHIELD_CHANCE
        self.boost = pod.boost_available and not self.shield and randint(0, 100) < BOOST_CHANCE

        pmin = self.thrust - 100 * amplitude
        pmax = self.thrust + 200 * amplitude

        if pmin < MIN_THRUST:
            pmin = MIN_THRUST
        elif pmin > 100:
            pmin = 100

        if pmax > 100:
            pmax = 100
        elif pmax < MIN_THRUST:
            pmax = MIN_THRUST

        if pmin <= pmax:
            self.thrust = uniform(pmin, pmax)
        else:
            self.thrust = uniform(pmin, pmax)

    @staticmethod
    def cross(p1_move, p2_move, pod):
        proba = randint(0, 100)

        if proba < 50:
            thrust = (0.7 * p1_move.thrust) + (0.3 * p2_move.thrust)
            move = Move((0.7 * p1_move.angle) + (0.3 * p2_move.angle), thrust)

        else:
            thrust = (0.7 * p2_move.thrust) + (0.3 * p1_move.thrust)
            move = Move((0.7 * p2_move.angle) + (0.3 * p1_move.angle), thrust)

        if p1_move.shield and p2_move.shield and randint(0, 100) < SHIELD_CHANCE and pod.shield_ready:
            move.shield = True

        if p1_move.boost and p2_move.boost and randint(0,
                                                       100) < BOOST_CHANCE and pod.boost_available and not move.shield:
            move.boost = True

        return move


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

        self.__generate_population(True)

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
                        p_ag_parameters = individual.coefficients
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
        self.maximum = -inf
        self.average = 0.0

        for individual in self.population:
            individual.score()
            self.update_avg_max(individual.score)

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

        self.population.append(self.get_best_solution().clone())

        for i in range(self.population_size):

            individual = self.population[i]
            if self.apocalypse < self.apocalypse_threshold:
                mutation_probability = self.compute_probability_mutation(individual.score)

                if uniform(0.0, 1.0) <= mutation_probability:
                    individual.mutate(mutation_probability)
                    individual.score()
                    new_average += individual.score
            else:
                individual.mutate(self.apocalypse_mutation_factor, True)
                individual.score()
                new_average += individual.score

            child = None
            if uniform(0.0, 1.0) <= crossing_probability:  # and nb_children <= MAX_NB_CHILDREN:
                child = self.crossing_mutation_single()
                child.score()
                self.population.append(child)
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
        return self.population[0]

    def generate_initial_population(self):
        '''
        Generate the population of solutions
        The populations is sorted with the best solution first
        @param is_first_generation: True if the generation is the really first one
        @return: None
        '''

        #TODO: adapt to Tron

        ref_individual = Individual(Individual.get_actual_coefficients())
        self.population.append(ref_individual)

        for i in range(self.population_size - 1):
            individual = Individual()
            individual.generate_individual_from_reference(ref_individual)
            self.population.append(individual)