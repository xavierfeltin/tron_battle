import threading
import gc
from GameFrame import GameFrame
from GameEngine import GameEngine
from RandomBot import RandomBot
from ExplicitBot import ExplicitBot
from MCTSBot import MCTSBot
from time import sleep

NB_PLAYERS = 2

def game_loop(engine, main_frame):
    gc.disable()
    while main_frame.is_running():

        if engine.is_game_playing():
            engine.update()
        else:
            main_frame.display_game_over()

        main_frame.refresh()
        main_frame.update_idletasks()
        main_frame.update()

        sleep(0.2)
    gc.enable()

if __name__ == "__main__":
    engine = GameEngine(NB_PLAYERS)

    players = []
    players.append(MCTSBot())
    for i in range(NB_PLAYERS-1):
        players.append(ExplicitBot())

    main_frame = GameFrame(engine)

    engine.initialize(players)
    main_frame.initialize()
    game_loop(engine, main_frame)