import threading
from GameFrame import GameFrame
from GameEngine import GameEngine


NB_PLAYERS = 3

def game_loop(engine, main_frame):
    while engine.is_game_playing():
        engine.update()

        main_frame.refresh()
        main_frame.update_idletasks()
        main_frame.update()

        for i in range(NB_PLAYERS):
            print(str(i) + ': ' + str(engine.cycles_positions[i]), flush=True)

if __name__ == "__main__":
    engine = GameEngine(NB_PLAYERS)
    main_frame = GameFrame(engine)

    engine.initialize()
    main_frame.initialize()
    game_loop(engine, main_frame)