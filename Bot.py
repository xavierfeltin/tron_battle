class Bot():
    def __init__(self):
        self.position = (-1, -1)
        self.wall = [] #list of previous positions

    def initialize(self, position):
        self.position = position
        self.wall.append(position)

    def compute_direction(self):
        pass
