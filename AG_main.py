import threading
import gc
from random import randint
from AG import AG
from time import sleep

if __name__ == "__main__":
    ag = AG()
    coefficients = ag.run()
    print('best coefficients = ' + str(coefficients))

