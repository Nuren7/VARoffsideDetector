import unittest

import cv2
import numpy as np

from app.ai.geometry import freeze_offside_snapshots, project_ground_positions
from app.ai.pitch import PitchCalibrator


class PitchCoordinateTests(unittest.TestCase):
    def setUp(self):
        self.image_points = [
            [100, 100],
            [900, 100],
            [1000, 700],
            [0, 700],
        ]
        self.pitch_points = [
            [0, 0],
            [105, 0],
            [105, 68],
            [0, 68],
        ]
        self.homography = PitchCalibrator._find_homography(
            self.image_points, self.pitch_points
        )

    def test_homography_maps_image_point_to_pitch(self):
        mapped = PitchCalibrator.to_pitch([[100, 100]], self.homography)[0]
        np.testing.assert_allclose(mapped, [0, 0], atol=0.1)

    def test_ground_points_are_projected_for_players_and_ball(self):
        video_data = [
            {
                "players": [{"bbox": [490, 380, 510, 400], "track_id": 1}],
                "ball": {"bbox": [495, 390, 505, 400]},
            }
        ]
        projected = project_ground_positions(
            video_data, [self.homography.tolist()]
        )
        self.assertIn("pitch_xy", projected[0]["players"][0])
        self.assertIn("pitch_xy", projected[0]["ball"])

    def test_pitch_line_draws_in_perspective(self):
        frame = np.zeros((800, 1100, 3), dtype=np.uint8)
        PitchCalibrator.draw_pitch_line(
            frame, 52.5, 68, self.homography
        )
        yellow_pixels = np.all(frame == (0, 255, 255), axis=2)
        self.assertGreater(int(yellow_pixels.sum()), 100)

    def test_player_attack_direction_selects_frozen_snapshot_at_release(self):
        frames = []
        for ball_x in (50, 51, 55, 60, 55):
            frames.append(
                {
                    "pitch_homography": self.homography.tolist(),
                    "players": [
                        {
                            "bbox": [0, 0, 10, 20],
                            "pitch_xy": [50, 34],
                            "track_id": 1,
                        },
                        {
                            "bbox": [0, 0, 10, 20],
                            "pitch_xy": [40, 34],
                            "track_id": 2,
                        },
                        {
                            "bbox": [0, 0, 10, 20],
                            "pitch_xy": [80, 34],
                            "track_id": 3,
                        },
                        {
                            "bbox": [0, 0, 10, 20],
                            "pitch_xy": [60, 34],
                            "track_id": 4,
                        },
                    ],
                    "ball": {
                        "bbox": [0, 0, 2, 2],
                        "pitch_xy": [ball_x, 34],
                    },
                }
            )

        snapshots = freeze_offside_snapshots(
            frames,
            [{0: 0, 1: 0, 2: 1, 3: 1} for _ in frames],
            {0: "right", 1: "left"},
            105,
            500,
            True,
        )
        self.assertEqual(snapshots[2]["attacking_team"], 0)
        self.assertEqual(snapshots[2]["line_x"], 60)
        self.assertEqual(snapshots[4]["attacking_team"], 1)
        self.assertEqual(snapshots[4]["line_x"], 50)


if __name__ == "__main__":
    unittest.main()
