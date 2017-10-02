import threading
from GameFrame import GameFrame
from GameEngine import GameEngine
from RandomBot import RandomBot
from ExplicitBot import ExplicitBot
from time import sleep

NB_PLAYERS = 3

def game_loop(engine, main_frame):
    while main_frame.is_running():

        if engine.is_game_playing():
            engine.update()

            for i in range(NB_PLAYERS):
                print(str(i) + ': ' + str(engine.cycles_positions[i]), flush=True)
        else:
            main_frame.display_game_over()

        main_frame.refresh()
        main_frame.update_idletasks()
        main_frame.update()

        sleep(0.5)

if __name__ == "__main__":
    engine = GameEngine(NB_PLAYERS)

    players = []
    for i in range(NB_PLAYERS):
        players.append(ExplicitBot())

    main_frame = GameFrame(engine)

    engine.initialize(players)
    main_frame.initialize()
    game_loop(engine, main_frame)