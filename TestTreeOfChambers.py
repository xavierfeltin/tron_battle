import unittest
import Configuration
from numpy import ones, zeros, copy, int32, sign
from TreeOfChambers import is_articulation_point

class TestTreeOfChambers(unittest.TestCase):
    def test_not_articulation_point(self):
        '''
        The considered point is not supposed to be an articulation point (more than one freedom degree)
        '''
        area = zeros((Configuration.MAX_X_GRID+1, Configuration.MAX_Y_GRID+1), dtype=bool)
        is_it = is_articulation_point(area, (2,2), [0, 1])
        self.assertFalse(is_it)

    def test_articulation_point(self):
        '''
        The considered point is an articulation point (one freedom degree remaining)
        '''
        area = zeros((Configuration.MAX_X_GRID + 1, Configuration.MAX_Y_GRID + 1), dtype=bool)
        area[1,1] = 1

        is_it = is_articulation_point(area, (1, 0), [-1, 0])
        self.assertTrue(is_it)

    def test_not_articulation_point_dead_end(self):
        '''
        The considered point is not an articulation point since it is a dead end (no more freedom degree remaining)
        '''
        area = zeros((Configuration.MAX_X_GRID + 1, Configuration.MAX_Y_GRID + 1), dtype=bool)
        area[1,1] = 1
        area[2,0] = 1

        is_it = is_articulation_point(area, (1, 0), [-1, 0])
        self.assertFalse(is_it)