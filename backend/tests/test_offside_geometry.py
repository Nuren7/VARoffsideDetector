import unittest

from app.ai.geometry import (
    calculate_offside_line,
    is_offside_position,
    smooth_line_positions,
    smooth_video_detections,
    stabilize_team_sequence,
)


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

    def test_pitch_coordinates_use_footpoint_and_team_attack_direction(self):
        pitch_players = [
            {"bbox": [0, 0, 20, 40], "pitch_xy": [80, 34]},
            {"bbox": [0, 0, 20, 40], "pitch_xy": [60, 34]},
            {"bbox": [0, 0, 20, 40], "pitch_xy": [50, 34]},
        ]
        pitch_ball = {"bbox": [0, 0, 10, 10], "pitch_xy": [55, 34]}
        line_right = calculate_offside_line(
            pitch_players, {0: 0, 1: 0, 2: 1}, pitch_ball, 0, "right"
        )
        self.assertEqual(line_right, 60)
        self.assertTrue(is_offside_position(70, line_right, 52.5, "right"))

        line_left = calculate_offside_line(
            pitch_players, {0: 0, 1: 0, 2: 1}, pitch_ball, 0, "left"
        )
        self.assertEqual(line_left, 55)
        self.assertTrue(is_offside_position(40, line_left, 52.5, "left"))

    def test_isolated_team_guess_does_not_flip_team(self):
        self.assertEqual(
            stabilize_team_sequence([0, 0, 1, 0, 0]),
            [0, 0, 0, 0, 0],
        )

    def test_line_median_removes_single_frame_spike_and_fills_short_gap(self):
        self.assertEqual(
            smooth_line_positions([600, 602, 760, 604, 605]),
            [602.0, 603.0, 604.0, 604.5, 605.0],
        )
        self.assertEqual(
            smooth_line_positions([600, None, 604]),
            [602.0, 602.0, 602.0],
        )

    def test_tracked_player_box_uses_temporal_median(self):
        video_data = [
            {"players": [{"bbox": [x, 0, x + 20, 40], "track_id": 7}]}
            for x in [10, 11, 100, 12, 13]
        ]
        smoothed = smooth_video_detections(video_data)
        self.assertEqual(smoothed[2]["players"][0]["bbox"], [12, 0, 32, 40])
        self.assertEqual(video_data[2]["players"][0]["bbox"], [100, 0, 120, 40])


if __name__ == "__main__":
    unittest.main()
