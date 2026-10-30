import unittest

from app.ai.geometry import calculate_offside_line, is_offside_position


class OffsideGeometryTests(unittest.TestCase):
    def setUp(self):
        self.players = [
            {"bbox": [700, 100, 750, 300]},
            {"bbox": [600, 100, 650, 300]},
            {"bbox": [500, 100, 550, 300]},
        ]
        self.teams = {0: 0, 1: 0, 2: 1}

    def test_rightward_line_uses_second_last_defender_and_ball(self):
        ball = {"bbox": [520, 200, 530, 210]}
        self.assertEqual(
            calculate_offside_line(self.players, self.teams, ball, 0, "right"),
            650,
        )

    def test_ball_can_be_the_more_restrictive_line(self):
        ball = {"bbox": [680, 200, 690, 210]}
        self.assertEqual(
            calculate_offside_line(self.players, self.teams, ball, 0, "right"),
            685,
        )

    def test_leftward_line_uses_second_last_defender_and_ball(self):
        ball = {"bbox": [620, 200, 630, 210]}
        self.assertEqual(
            calculate_offside_line(self.players, self.teams, ball, 0, "left"),
            625,
        )

    def test_line_is_not_drawable_without_ball_or_two_opponents(self):
        ball = {"bbox": [520, 200, 530, 210]}
        self.assertIsNone(
            calculate_offside_line(self.players, self.teams, None, 0, "right")
        )
        self.assertIsNone(
            calculate_offside_line(
                self.players, {0: 0, 1: 1, 2: 1}, ball, 0, "right"
            )
        )

    def test_equal_line_and_midfield_positions_are_onside(self):
        self.assertFalse(is_offside_position([640, 0, 650, 20], 650, 500, "right"))
        self.assertFalse(is_offside_position([490, 0, 500, 20], 450, 500, "right"))
        self.assertFalse(is_offside_position([450, 0, 500, 20], 450, 500, "left"))
        self.assertFalse(is_offside_position([500, 0, 550, 20], 550, 500, "left"))

    def test_requires_player_to_be_ahead_of_both_line_and_midfield(self):
        self.assertTrue(is_offside_position([660, 0, 670, 20], 650, 500, "right"))
        self.assertFalse(is_offside_position([660, 0, 670, 20], 680, 500, "right"))
        self.assertTrue(is_offside_position([330, 0, 340, 20], 350, 500, "left"))


if __name__ == "__main__":
    unittest.main()
