from typing import Dict, List

import cv2
import numpy as np
from sklearn.cluster import KMeans


class Teamclassifier:
    """Assign consistent team IDs by clustering jersey colors across a full video."""

    def __init__(self, n_clusters: int = 2):
        if n_clusters != 2:
            raise ValueError("Team classification requires exactly two team clusters.")
        self.n_clusters = n_clusters

    def extract_jersey_color(self, frame: np.ndarray, bbox: list) -> np.ndarray:
        """Return a robust Lab color feature from the central upper-body crop."""
        x1, y1, x2, y2 = map(int, bbox)
        height, width = frame.shape[:2]
        box_width = x2 - x1
        box_height = y2 - y1
        if box_width <= 0 or box_height <= 0:
            return np.array([])

        crop_x1 = max(0, x1 + int(box_width * 0.2))
        crop_x2 = min(width, x2 - int(box_width * 0.2))
        crop_y1 = max(0, y1 + int(box_height * 0.12))
        crop_y2 = min(height, y1 + int(box_height * 0.62))
        if crop_x1 >= crop_x2 or crop_y1 >= crop_y2:
            return np.array([])

        crop = frame[crop_y1:crop_y2, crop_x1:crop_x2]
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        saturated_pixels = hsv[hsv[:, :, 1] >= 35]
        if len(saturated_pixels) >= 5:
            lab_pixels = cv2.cvtColor(
                saturated_pixels.reshape(-1, 1, 3), cv2.COLOR_HSV2BGR
            )
            lab_pixels = cv2.cvtColor(lab_pixels, cv2.COLOR_BGR2LAB).reshape(-1, 3)
        else:
            lab_pixels = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB).reshape(-1, 3)

        return np.median(lab_pixels, axis=0).astype(np.float32)

    def classify_video(
        self, video_path: str, video_data: List[Dict]
    ) -> List[Dict[int, int]]:
        """Fit both team colors together and return stable labels per frame."""
        features: List[np.ndarray] = []
        locations: List[tuple[int, int]] = []
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")

        frame_index = 0
        try:
            while True:
                success, frame = cap.read()
                if not success:
                    break

                if frame_index < len(video_data):
                    for player_index, player in enumerate(
                        video_data[frame_index].get("players", [])
                    ):
                        feature = self.extract_jersey_color(frame, player["bbox"])
                        if feature.size:
                            features.append(feature)
                            locations.append((frame_index, player_index))
                frame_index += 1
        finally:
            cap.release()

        frame_teams = [dict() for _ in video_data]
        if len(features) < self.n_clusters:
            return frame_teams

        feature_array = np.asarray(features)
        kmeans = KMeans(n_clusters=self.n_clusters, n_init=10, random_state=42)
        labels = kmeans.fit_predict(feature_array)
        if np.linalg.norm(kmeans.cluster_centers_[0] - kmeans.cluster_centers_[1]) < 8:
            return frame_teams

        for (frame_index, player_index), label in zip(locations, labels):
            frame_teams[frame_index][player_index] = int(label)
        return frame_teams
