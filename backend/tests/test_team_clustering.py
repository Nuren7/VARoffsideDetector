import os
import tempfile
import unittest

import cv2
import numpy as np

from app.ai.clustering import Teamclassifier


class TeamClusteringTests(unittest.TestCase):
    def test_team_ids_are_stable_across_video_frames(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            video_path = os.path.join(temp_dir, "teams.avi")
            writer = cv2.VideoWriter(
                video_path,
                cv2.VideoWriter_fourcc(*"MJPG"),
                5,
                (160, 100),
            )
            if not writer.isOpened():
                self.skipTest("OpenCV MJPG video writer is unavailable.")

            frame_data = []
            for _ in range(4):
                frame = np.full((100, 160, 3), (40, 120, 40), dtype=np.uint8)
                frame[25:60, 28:52] = (0, 0, 220)
                frame[25:60, 108:132] = (220, 0, 0)
                writer.write(frame)
                frame_data.append(
                    {
                        "players": [
                            {"bbox": [20, 10, 60, 90]},
                            {"bbox": [100, 10, 140, 90]},
                        ]
                    }
                )
            writer.release()

            assignments = Teamclassifier().classify_video(video_path, frame_data)

        self.assertEqual(len(assignments), 4)
        first_frame = assignments[0]
        self.assertEqual(set(first_frame), {0, 1})
        self.assertNotEqual(first_frame[0], first_frame[1])
        for frame_assignments in assignments[1:]:
            self.assertEqual(frame_assignments, first_frame)


if __name__ == "__main__":
    unittest.main()
