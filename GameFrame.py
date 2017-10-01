from tkinter import *

WIDTH = 600
HEIGHT = 400
X_GRID_OFFSET = 20
Y_GRID_OFFSET = 20

COLORS = ['red', 'yellow', 'blue', 'green']

class GameFrame(Tk):

    def __init__(self, game_engine):
        Tk.__init__(self)

        self.engine = game_engine

        # Add a canvas to frame1 as self.canvas member
        self.canvas = Canvas(self, width=WIDTH, height=HEIGHT, bg="black")
        self.canvas.focus_set()
        self.canvas.pack()
        self.initialize()

    def initialize(self):
        '''
        Initialize the drawing area
        '''
        self.canvas.delete(ALL)
        self.refresh()

    def refresh(self):
        '''
        Draw the modifications on the screen
        '''

        for index, position in enumerate(self.engine.cycles_positions):
            # Initialize cycle with 1 rectangle at its current position
            x0 = position[0] * X_GRID_OFFSET
            y0 = position[1] * Y_GRID_OFFSET
            x1 = position[0] * X_GRID_OFFSET + X_GRID_OFFSET
            y1 = position[1] * Y_GRID_OFFSET + Y_GRID_OFFSET
            tag_rect = 'rect' + str(index)

            self.canvas.create_rectangle(x0, y0, x1, y1, outline=COLORS[index], fill=COLORS[index])



