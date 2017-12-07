import threading
import gc
from random import randint
from AG import AG
from time import sleep
from math import inf

if __name__ == "__main__":
    ag = AG(population_size = 50, nb_games = 10, min_evaluations = 300, nb_tournament = 30, nb_tournament_contestants= 2, apocalypse_threshold = inf, apocalypse_mutation_factor = 0.5)
    #ag = AG()
    coefficients = ag.run()
    print('best coefficients = ' + str(coefficients))

    file = open("coefficients.txt", "w")
    file.write(str(coefficients))
    file.close()

