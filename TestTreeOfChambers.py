import unittest
import Configuration
from numpy import ones, zeros, copy, int32, sign
from TreeOfChambers import compute_tree_of_chambers, detect_articulation_points

class TestTreeOfChambers(unittest.TestCase):

    def test_empty_list_articulation_points(self):
        '''
        No articulation point present in the graph
        '''
        area = zeros((Configuration.MAX_X_GRID + 1, Configuration.MAX_Y_GRID + 1), dtype=bool)
        points = detect_articulation_points(area, (2,2))
        self.assertEqual(len(points), 0)

    def test_wall_list_articulation_points(self):
        '''
        3 articulations points expected when the board is split in two with a wall
        '''
        area = zeros((Configuration.MAX_X_GRID + 1, Configuration.MAX_Y_GRID + 1), dtype=bool)

        for i in range(1,Configuration.MAX_Y_GRID+1):
            area[4, i] = 1

        points = detect_articulation_points(area, (0, 0))
        self.assertEqual(len(points), 3)

        expected_articulations = [(3,0), (4,0), (5,0)]
        for articulation in points:
            self.assertTrue(articulation in expected_articulations)

    def test_small_hall_list_articulation_points(self):
        '''
        0 articulations points expected when a small hall is on the map
        '''
        area = zeros((Configuration.MAX_X_GRID + 1, Configuration.MAX_Y_GRID + 1), dtype=bool)

        for i in range(5, 8):
            area[i, 3] = 1
            area[i, 5] = 1

        points = detect_articulation_points(area, (0, 0))
        self.assertEqual(len(points), 0)

    def test_chamber_list_articulation_points(self):
        '''
        3 articulations points expected when a chamber is on the map
        '''
        area = zeros((Configuration.MAX_X_GRID + 1, Configuration.MAX_Y_GRID + 1), dtype=bool)
        for i in range(0, 8): area[3, i] = 1
        for i in range(3, 10): area[i, 7] = 1
        for i in range(0, 3): area[9, i] = 1
        for i in range(4, 10): area[9, i] = 1

        points = detect_articulation_points(area, (9, 2))
        self.assertEqual(len(points), 3)

        expected_articulations = [(8, 3), (9, 3), (10, 3)]
        for articulation in points:
            self.assertTrue(articulation in expected_articulations)

    def test_chamber_hall_list_articulation_points(self):
        '''
        3 articulations points expected when a chamber and a hall is on the map
        '''
        area = zeros((Configuration.MAX_X_GRID + 1, Configuration.MAX_Y_GRID + 1), dtype=bool)
        for i in range(1, Configuration.MAX_Y_GRID+1): area[3, i] = 1
        for i in range(3, 10):
            area[i, Configuration.MAX_Y_GRID] = 1
            area[i, 1] = 1
        for i in range(1, 3): area[9, i] = 1
        for i in range(4, Configuration.MAX_Y_GRID+1): area[9, i] = 1

        points = detect_articulation_points(area, (9, 2))
        self.assertEqual(len(points), 12)

        expected_articulations = [(2, 0), (3, 0), (4, 0), (5, 0), (6, 0), (7, 0), (8, 0), (9, 0), (10, 0), (8, 3), (9, 3), (10, 3)]
        for articulation in points:
            self.assertTrue(articulation in expected_articulations)

    def test_complex_list_articulation_points(self):
        '''
        from: https://project.dke.maastrichtuniversity.nl/games/files/bsc/Kang_Bsc-paper.pdf(Fig 6.)
        X articulations points expected when a chamber and a hall is on the map
        '''
        area = zeros((Configuration.MAX_X_GRID + 1, Configuration.MAX_Y_GRID + 1), dtype=bool)
        wall_spaces = [(1, 1),(2, 1),(6, 1),(8, 1),(10, 1),(11, 1)
            ,(1, 2),(6, 2),(8, 2),(11, 2)
            ,(8, 3)
            ,(5, 4),(6, 4),(7, 4),(8, 4),(9, 4),(10, 4)
            ,(1, 5),(2, 5),(3, 5),(4, 5),(9, 5),(10,5)
            ,(1, 6),(2, 6),(3, 6),(4, 6),(9, 6),(10,6),(11,6)
            ,(3, 7),(9, 7),(10, 7),(11, 7)
            ,(3, 8),(5, 8),(6, 8),(7, 8)
            ,(3, 9),(4, 9),(5, 9)
            ,(1, 10),(5, 10),(6, 10),(11, 10)
            ,(1, 11),(2, 11),(6, 11),(10, 11),(11, 11)]

        for wall in wall_spaces:
            area[wall] = 1

        for i in range(0, Configuration.MAX_X_GRID+1):
            for j in range(13, Configuration.MAX_Y_GRID+1):
                area[i,j] = 1

        for i in range(13, Configuration.MAX_X_GRID+1):
            for j in range(0, 13):
                area[i,j] = 1

        points = detect_articulation_points(area, (0, 0))
        self.assertEqual(len(points), 5)

        expected_articulations = [(7, 9), (4, 7), (5, 7), (8, 7), (8, 8)]
        for articulation in points:
            self.assertTrue(articulation in expected_articulations)

    def test_compute_tree_of_chambers_empty(self):
        area = zeros((Configuration.MAX_X_GRID + 1, Configuration.MAX_Y_GRID + 1), dtype=bool)
        voronoi = ones((Configuration.MAX_X_GRID+1, Configuration.MAX_Y_GRID+1), dtype=int32)
        current_pos = (15,15)

        articulations = detect_articulation_points(area, current_pos)
        nb_spaces = compute_tree_of_chambers(area, voronoi, articulations, current_pos, (-1,-1))

        self.assertEqual(nb_spaces, (Configuration.MAX_X_GRID+1) * (Configuration.MAX_Y_GRID+1))

    def test_compute_tree_of_one_chamber(self):
        area = zeros((Configuration.MAX_X_GRID + 1, Configuration.MAX_Y_GRID + 1), dtype=bool)
        for i in range(0, 8): area[3, i] = 1
        for i in range(3, 10): area[i, 7] = 1
        for i in range(0, 3): area[9, i] = 1
        for i in range(4, 8): area[9, i] = 1

        voronoi = ones((Configuration.MAX_X_GRID + 1, Configuration.MAX_Y_GRID + 1), dtype=int32)
        current_pos = (15, 15)

        articulations = detect_articulation_points(area, current_pos)
        nb_spaces = compute_tree_of_chambers(area, voronoi, articulations, current_pos, (-1, -1))

        self.assertEqual(nb_spaces, 577)

    def test_compute_tree_of_choice_chamber(self):
        area = zeros((Configuration.MAX_X_GRID + 1, Configuration.MAX_Y_GRID + 1), dtype=bool)
        for i in range(1, 14): area[i, 6] = 1
        for i in range(7, 19): area[1, i] = 1
        for i in range(7, 19): area[13, i] = 1
        for i in range(2, 6): area[i, 18] = 1
        for i in range(10, 19): area[5, i] = 1
        for i in range(6, 11): area[i, 10] = 1
        for i in range(8, 19): area[10, i] = 1
        for i in range(11, 14): area[i, 18] = 1
        area[12, 7] = 1

        voronoi = ones((Configuration.MAX_X_GRID + 1, Configuration.MAX_Y_GRID + 1), dtype=int32)
        current_pos = (11, 7)

        articulations = detect_articulation_points(area, current_pos)
        nb_spaces = compute_tree_of_chambers(area, voronoi, articulations, current_pos, (7, 12))

        self.assertEqual(nb_spaces, 48)

    def test_compute_tree_of_complex_choice_chamber(self):
        area = zeros((Configuration.MAX_X_GRID + 1, Configuration.MAX_Y_GRID + 1), dtype=bool)
        for i in range(0, 11): area[i, 5] = 1
        for i in range(12, 23): area[i, 5] = 1
        for i in range(0, 13): area[i, 9] = 1
        for i in range(1, 6): area[4, i] = 1
        for i in range(1, 6): area[10, i] = 1
        for i in range(1, 12): area[12, i] = 1
        for i in range(0, 6): area[22, i] = 1

        voronoi = ones((Configuration.MAX_X_GRID + 1, Configuration.MAX_Y_GRID + 1), dtype=int32)
        current_pos = (11, 0)

        articulations = detect_articulation_points(area, current_pos)
        nb_spaces = compute_tree_of_chambers(area, voronoi, articulations, current_pos, (-1, -1))

        self.assertEqual(nb_spaces, 45)