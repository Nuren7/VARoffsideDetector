import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import cv2
import numpy as np
from ultralytics import YOLO


class PitchCalibrator:
    """Map image points to standard pitch coordinates using a homography."""

    def __init__(
        self,
        calibration_path: Optional[str] = None,
        model_path: Optional[str] = None,
        keypoint_pitch_points: Optional[Dict[int, Sequence[float]]] = None,
        pitch_length_m: float = 105.0,
        pitch_width_m: float = 68.0,
        keypoint_confidence: float = 0.35,
        smoothing: float = 0.25,
    ):
        if pitch_length_m <= 0 or pitch_width_m <= 0:
            raise ValueError("Pitch dimensions must be positive.")
        if not 0 <= smoothing <= 1:
            raise ValueError("Homography smoothing must be between 0 and 1.")
        self.pitch_length_m = pitch_length_m
        self.pitch_width_m = pitch_width_m
        self.keypoint_confidence = keypoint_confidence
        self.smoothing = smoothing
        self._previous_homography: Optional[np.ndarray] = None
        self._missing_frames = 0
        self._manual_homography: Optional[np.ndarray] = None
        self._model = YOLO(model_path) if model_path else None
        self._keypoint_pitch_points = keypoint_pitch_points or {}

        if calibration_path:
            path = Path(calibration_path)
            if not path.is_file():
                raise FileNotFoundError(f"Pitch calibration file not found: {path}")
            with path.open(encoding="utf-8") as calibration_file:
                calibration = json.load(calibration_file)
            self.pitch_length_m = float(
                calibration.get("pitch_length_m", pitch_length_m)
            )
            self.pitch_width_m = float(
                calibration.get("pitch_width_m", pitch_width_m)
            )
            image_points = calibration.get("image_points")
            pitch_points = calibration.get("pitch_points_m")
            if image_points is not None or pitch_points is not None:
                if image_points is None or pitch_points is None:
                    raise ValueError(
                        "Manual calibration must include both image_points and "
                        "pitch_points_m."
                    )
                self._manual_homography = self._find_homography(
                    image_points, pitch_points
                )
            configured_keypoints = calibration.get("keypoint_pitch_points_m")
            if configured_keypoints is not None:
                self._keypoint_pitch_points = {
                    int(index): point
                    for index, point in configured_keypoints.items()
                }

        if self._model is not None and len(self._keypoint_pitch_points) < 4:
            raise ValueError(
                "Pitch keypoint detection needs at least four keypoint-to-pitch "
                "coordinate mappings."
            )
        if self._model is None and self._manual_homography is None:
            raise ValueError(
                "Pitch calibration needs manual image/pitch points or a pose model "
                "with keypoint-to-pitch mappings."
            )

    @staticmethod
    def _find_homography(
        image_points: Sequence[Sequence[float]],
        pitch_points: Sequence[Sequence[float]],
    ) -> np.ndarray:
        image_array = np.asarray(image_points, dtype=np.float32)
        pitch_array = np.asarray(pitch_points, dtype=np.float32)
        if (
            image_array.ndim != 2
            or image_array.shape[1] != 2
            or pitch_array.shape != image_array.shape
            or len(image_array) < 4
        ):
            raise ValueError(
                "Homography needs at least four matching image and pitch points."
            )

        homography, inliers = cv2.findHomography(
            image_array, pitch_array, cv2.RANSAC, 3.0
        )
        if homography is None or inliers is None or int(inliers.sum()) < 4:
            raise ValueError(
                "Could not compute a valid pitch homography from the supplied points."
            )
        normalized = PitchCalibrator._normalize_homography(homography)
        if normalized is None:
            raise ValueError("The supplied points produce a singular homography.")
        return normalized

    @staticmethod
    def _normalize_homography(
        homography: np.ndarray,
    ) -> Optional[np.ndarray]:
        if (
            not np.isfinite(homography).all()
            or abs(float(homography[2, 2])) < 1e-10
        ):
            return None
        normalized = homography / homography[2, 2]
        if abs(float(np.linalg.det(normalized))) < 1e-12:
            return None
        return normalized

    @staticmethod
    def _smooth_homography(
        previous: Optional[np.ndarray], current: np.ndarray, weight: float
    ) -> np.ndarray:
        current = PitchCalibrator._normalize_homography(current)
        if current is None:
            raise ValueError("Invalid pitch homography.")
        if previous is None:
            return current
        smoothed = (1 - weight) * previous + weight * current
        normalized = PitchCalibrator._normalize_homography(smoothed)
        return normalized if normalized is not None else current

    def homography_for_frame(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """Return image-to-pitch homography; recalibrate each frame when configured."""
        homography = self._manual_homography
        if self._model is not None:
            result = self._model(frame, verbose=False)[0]
            keypoints = result.keypoints
            if keypoints is None or len(keypoints) == 0:
                return self._reuse_recent_homography()

            pose_index = 0
            if result.boxes is not None and len(result.boxes) > 0:
                pose_index = int(result.boxes.conf.argmax().item())

            image_points = keypoints.xy[pose_index].cpu().numpy()
            confidence = (
                keypoints.conf[pose_index].cpu().numpy()
                if keypoints.conf is not None
                else np.ones(len(image_points), dtype=np.float32)
            )
            matched_image_points = []
            matched_pitch_points = []
            for keypoint_index, pitch_point in self._keypoint_pitch_points.items():
                if (
                    keypoint_index < len(image_points)
                    and confidence[keypoint_index] >= self.keypoint_confidence
                ):
                    matched_image_points.append(image_points[keypoint_index])
                    matched_pitch_points.append(pitch_point)
            if len(matched_image_points) < 4:
                return self._reuse_recent_homography()

            homography, inliers = cv2.findHomography(
                np.asarray(matched_image_points, dtype=np.float32),
                np.asarray(matched_pitch_points, dtype=np.float32),
                cv2.RANSAC,
                3.0,
            )
            if homography is None or inliers is None or int(inliers.sum()) < 4:
                return self._reuse_recent_homography()
            homography = self._normalize_homography(homography)
            if homography is None:
                return self._reuse_recent_homography()

        if homography is None:
            return None

        self._missing_frames = 0
        self._previous_homography = self._smooth_homography(
            self._previous_homography, homography, self.smoothing
        )
        return self._previous_homography.copy()

    def _reuse_recent_homography(self, max_missing_frames: int = 5) -> Optional[np.ndarray]:
        self._missing_frames += 1
        if (
            self._previous_homography is not None
            and self._missing_frames <= max_missing_frames
        ):
            return self._previous_homography.copy()
        return None

    @staticmethod
    def to_pitch(
        points_xy: Sequence[Sequence[float]], homography: np.ndarray
    ) -> np.ndarray:
        points = np.asarray(points_xy, dtype=np.float32).reshape(-1, 1, 2)
        return cv2.perspectiveTransform(points, homography).reshape(-1, 2)

    @staticmethod
    def draw_pitch_line(
        frame: np.ndarray,
        x_pitch: float,
        pitch_width_m: float,
        homography: np.ndarray,
        color: tuple[int, int, int] = (0, 255, 255),
    ) -> None:
        inverse = np.linalg.inv(homography)
        endpoints = PitchCalibrator.to_pitch(
            [(x_pitch, 0), (x_pitch, pitch_width_m)], inverse
        )
        height, width = frame.shape[:2]
        p1 = tuple(np.rint(endpoints[0]).astype(int))
        p2 = tuple(np.rint(endpoints[1]).astype(int))
        clipped, clipped_p1, clipped_p2 = cv2.clipLine(
            (0, 0, width, height), p1, p2
        )
        if clipped:
            cv2.line(frame, clipped_p1, clipped_p2, color, 2)
