import time
import cv2
import numpy as np
from typing import Any, Dict, List, Optional

from app.ai.progress import report_frame_progress
from sklearn.cluster import KMeans


def calculate_offside_line(
    players: List[Dict[str, Any]],
    player_teams: Dict[int, int],
    ball: Optional[Dict[str, Any]],
    defending_team: int,
    attacking_direction: str,
) -> Optional[float]:
    """Return the stricter of the ball and second-last-opponent positions."""
    if ball is None or attacking_direction not in {"left", "right"}:
        return None

    defender_edges = []
    for player_index, player in enumerate(players):
        if player_teams.get(player_index) != defending_team:
            continue
        x1, _, x2, _ = player["bbox"]

        defender_edges.append(x2 if attacking_direction == "right" else x1)

    if len(defender_edges) < 2:
        return None

    defender_edges.sort(reverse=attacking_direction == "right")
    second_last_defender = defender_edges[1]
    bx1, _, bx2, _ = ball["bbox"]
    ball_x = (bx1 + bx2) / 2
    return (
        max(second_last_defender, ball_x)
        if attacking_direction == "right"
        else min(second_last_defender, ball_x)
    )


def is_offside_position(
    bbox: List[float],
    line_x: float,
    midfield_x: float,
    attacking_direction: str,
) -> bool:
    """Approximate the position test using the leading edge of a player box."""
    if attacking_direction == "right":
        return bbox[2] > midfield_x and bbox[2] > line_x
    if attacking_direction == "left":
        return bbox[0] < midfield_x and bbox[0] < line_x
    raise ValueError("attacking_direction must be 'left' or 'right'.")


def smooth_video_detections(
    video_data: List[Dict[str, Any]], radius: int = 3
) -> List[Dict[str, Any]]:
    """Reduce detection jitter with an increased temporal median for each tracked player."""
    smoothed = [
        {
            **frame_data,
            "players": [dict(player) for player in frame_data.get("players", [])],
            "ball": (
                dict(frame_data["ball"])
                if frame_data.get("ball") is not None
                else None
            ),
        }
        for frame_data in video_data
    ]

    observations: Dict[int, Dict[int, List[float]]] = {}
    ball_observations: Dict[int, Dict[int, List[float]]] = {}
    for frame_index, frame_data in enumerate(video_data):
        for player in frame_data.get("players", []):
            track_id = player.get("track_id")
            if track_id is not None:
                observations.setdefault(int(track_id), {})[frame_index] = player[
                    "bbox"
                ]
        ball = frame_data.get("ball")
        if ball is not None and ball.get("track_id") is not None:
            ball_observations.setdefault(int(ball["track_id"]), {})[frame_index] = (
                ball["bbox"]
            )

    for frame_index, frame_data in enumerate(smoothed):
        for player in frame_data["players"]:
            track_id = player.get("track_id")
            if track_id is None:
                continue
            track_observations = observations[int(track_id)]
            nearby_boxes = [
                box
                for observed_frame, box in track_observations.items()
                if abs(observed_frame - frame_index) <= radius
            ]
            if nearby_boxes:
                player["bbox"] = np.median(nearby_boxes, axis=0).tolist()

        ball = frame_data.get("ball")
        if ball is not None:
            track_id = ball.get("track_id")
            if track_id is not None:
                track_observations = ball_observations[int(track_id)]
                nearby_boxes = [
                    box
                    for observed_frame, box in track_observations.items()
                    if abs(observed_frame - frame_index) <= radius
                ]
            else:
                nearby_boxes = [
                    video_data[nearby_frame]["ball"]["bbox"]
                    for nearby_frame in range(
                        max(0, frame_index - radius),
                        min(len(video_data), frame_index + radius + 1),
                    )
                    if video_data[nearby_frame].get("ball") is not None
                ]
            if nearby_boxes:
                ball["bbox"] = np.median(nearby_boxes, axis=0).tolist()

    return smoothed


def stabilize_team_sequence(
    team_ids: List[Optional[int]], radius: int = 3
) -> List[Optional[int]]:
    """Suppress isolated changes in the estimated team nearest the ball."""
    stabilized: List[Optional[int]] = []
    for frame_index, team_id in enumerate(team_ids):
        nearby = [
            candidate
            for candidate in team_ids[
                max(0, frame_index - radius) : min(
                    len(team_ids), frame_index + radius + 1
                )
            ]
            if candidate in {0, 1}
        ]
        if not nearby:
            stabilized.append(None)
            continue

        counts = {candidate: nearby.count(candidate) for candidate in {0, 1}}
        if counts[0] == counts[1] and team_id in {0, 1}:
            stabilized.append(team_id)
        else:
            stabilized.append(max(counts, key=counts.get))
    return stabilized


def smooth_line_positions(
    line_positions: List[Optional[float]], radius: int = 3, max_gap: int = 3, max_jump_delta: float = 25.0
) -> List[Optional[float]]:
    """Median-filter candidate lines, interpolate short gaps, and clamp sudden frame-to-frame jumps."""
    smoothed = list(line_positions)
    valid_indices = [
        index for index, value in enumerate(smoothed) if value is not None
    ]

    # 1. Interpolate short gaps between valid lines
    for left_index, right_index in zip(valid_indices, valid_indices[1:]):
        gap = right_index - left_index - 1
        if 0 < gap <= max_gap:
            for index in range(left_index + 1, right_index):
                ratio = (index - left_index) / (right_index - left_index)
                smoothed[index] = (
                    line_positions[left_index] * (1 - ratio)
                    + line_positions[right_index] * ratio
                )

    # 2. Apply rolling median filter to suppress high-frequency jitter
    original = list(smoothed)
    for frame_index, value in enumerate(original):
        if value is None:
            continue
        nearby = [
            candidate
            for candidate in original[
                max(0, frame_index - radius) : min(
                    len(original), frame_index + radius + 1
                )
            ]
            if candidate is not None
        ]
        if nearby:
            smoothed[frame_index] = float(np.median(nearby))

    # 3. Velocity clamping: Prevent the line from jumping too violently between frames
    for i in range(1, len(smoothed)):
        if smoothed[i] is not None and smoothed[i-1] is not None:
            diff = smoothed[i] - smoothed[i-1]
            if abs(diff) > max_jump_delta:
                smoothed[i] = smoothed[i-1] + np.sign(diff) * max_jump_delta

    return smoothed


class OffsideAnalyzer:
    """Draw a ball-relative candidate offside line and candidate attackers with maximum stability."""

    def __init__(
        self,
        output_video_path: str = "output_offside.mp4",
        fps: float = 30.0,
        attacking_direction: str = "right",
    ):
        if attacking_direction not in {"left", "right"}:
            raise ValueError("attacking_direction must be 'left' or 'right'.")
        self.output_video_path = output_video_path
        self.fps = fps
        self.attacking_direction = attacking_direction

    @staticmethod
    def _team_nearest_ball(
        players: List[Dict[str, Any]],
        player_teams: Dict[int, int],
        ball: Optional[Dict[str, Any]],
    ) -> Optional[int]:
        """Determine which team is closest to the ball in the current frame."""
        if ball is None:
            return None

        bx1, by1, bx2, by2 = ball["bbox"]
        ball_center = ((bx1 + bx2) / 2, (by1 + by2) / 2)
        nearest_distance = float("inf")
        nearest_team = None

        for player_index, player in enumerate(players):
            team_id = player_teams.get(player_index)
            if team_id not in {0, 1}:
                continue
            x1, _, x2, y2 = player["bbox"]
            foot_center = ((x1 + x2) / 2, y2)
            distance = np.hypot(
                ball_center[0] - foot_center[0], ball_center[1] - foot_center[1]
            )
            if distance < nearest_distance:
                nearest_distance = distance
                nearest_team = team_id

        return nearest_team

    def process_and_draw(
        self,
        input_video_path: str,
        video_data: List[Dict[str, Any]],
        player_teams: List[Dict[int, int]],
    ) -> None:
        """Process input video, apply multi-layer stabilization, and draw final annotations."""
        cap = cv2.VideoCapture(input_video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {input_video_path}")

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if video_fps <= 0 or np.isnan(video_fps):
            video_fps = self.fps

        # Use 'avc1' for browser and cloud storage compatibility
        fourcc = cv2.VideoWriter_fourcc(*"avc1")
        out = cv2.VideoWriter(self.output_video_path, fourcc, video_fps, (width, height))

        # Fallback to 'mp4v' if 'avc1' fails
        if not out.isOpened():
            out.release()
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out = cv2.VideoWriter(
                self.output_video_path, fourcc, video_fps, (width, height)
            )

        if not out.isOpened():
            cap.release()
            raise RuntimeError(
                f"Could not create output video: {self.output_video_path}"
            )

        # Apply multi-layer stabilization filters
        stable_video_data = smooth_video_detections(video_data)
        detected_attacking_teams = []
        for frame_index, frame_data in enumerate(stable_video_data):
            players = frame_data.get("players", [])
            ball = frame_data.get("ball")
            teams = (
                player_teams[frame_index]
                if frame_index < len(player_teams)
                else {}
            )
            attacking_team = self._team_nearest_ball(players, teams, ball)
            detected_attacking_teams.append(attacking_team)

        stable_attacking_teams = stabilize_team_sequence(detected_attacking_teams)
        candidate_lines = []
        for frame_index, frame_data in enumerate(stable_video_data):
            players = frame_data.get("players", [])
            ball = frame_data.get("ball")
            teams = (
                player_teams[frame_index]
                if frame_index < len(player_teams)
                else {}
            )
            attacking_team = stable_attacking_teams[frame_index]
            candidate_lines.append(
                calculate_offside_line(
                    players,
                    teams,
                    ball,
                    1 - attacking_team,
                    self.attacking_direction,
                )
                if attacking_team is not None
                else None
            )

        stable_lines = smooth_line_positions(candidate_lines)

        started_at = time.perf_counter()
        total_label = str(total_frames) if total_frames > 0 else "unknown"
        print(
            f"[processing] Video annotation started ({total_label} frames).",
            flush=True,
        )

        frame_index = 0
        try:
            while True:
                success, frame = cap.read()
                if not success:
                    break

                if frame_index < len(video_data):
                    frame_data = stable_video_data[frame_index]
                    players = frame_data.get("players", [])
                    ball = frame_data.get("ball")
                    teams = (
                        player_teams[frame_index]
                        if frame_index < len(player_teams)
                        else {}
                    )

                    # Draw bounding boxes and team labels for each player
                    for player_index, player in enumerate(players):
                        x1, y1, x2, y2 = map(int, player["bbox"])
                        team_id = teams.get(player_index, -1)
                        color = (
                            (0, 0, 255)
                            if team_id == 0
                            else (255, 0, 0)
                            if team_id == 1
                            else (0, 255, 255)
                        )
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                        if team_id in {0, 1}:
                            cv2.putText(
                                frame,
                                f"Team {team_id}",
                                (x1, max(y1 - 6, 15)),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.5,
                                color,
                                2,
                            )

                    attacking_team = stable_attacking_teams[frame_index]
                    line_x = stable_lines[frame_index]

                    # Draw the ball marker if present
                    if ball is not None:
                        bx1, by1, bx2, by2 = map(int, ball["bbox"])
                        center = ((bx1 + bx2) // 2, (by1 + by2) // 2)
                        cv2.rectangle(frame, (bx1, by1), (bx2, by2), (0, 255, 0), 2)
                        cv2.circle(frame, center, 5, (0, 255, 0), -1)

                    # Draw candidate offside line and check for offside positions
                    if line_x is not None:
                        current_line_x = int(round(line_x))
                        cv2.line(
                            frame,
                            (current_line_x, 0),
                            (current_line_x, height),
                            (0, 255, 255),
                            2,
                        )
                        cv2.putText(
                            frame,
                            "Candidate offside line",
                            (min(current_line_x + 5, width - 210), 30),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (0, 255, 255),
                            2,
                        )

                        for player_index, player in enumerate(players):
                            if teams.get(player_index) != attacking_team:
                                continue
                            if is_offside_position(
                                player["bbox"],
                                line_x,
                                width / 2,
                                self.attacking_direction,
                            ):
                                x1, y1, x2, y2 = map(int, player["bbox"])
                                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                                cv2.putText(
                                    frame,
                                    "Position candidate",
                                    (x1, max(y1 - 8, 15)),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.55,
                                    (0, 0, 255),
                                    2,
                                )

                out.write(frame)
                frame_index += 1
                report_frame_progress(
                    "Video annotation",
                    frame_index,
                    total_frames,
                    started_at,
                )
        finally:
            cap.release()
            out.release()

        report_frame_progress(
            "Video annotation",
            frame_index,
            total_frames,
            started_at,
            force=True,
        )
        print(
            f"[processing] Video annotation complete: {frame_index} frames written "
            f"to {self.output_video_path}.",
            flush=True,
        )


class Teamclassifier:
    """Assign stable team IDs by clustering jersey colors across a full video."""

    def __init__(self, n_clusters: int = 2):
        if n_clusters != 2:
            raise ValueError("Team classification requires exactly two team clusters.")
        self.n_clusters = n_clusters

    @staticmethod
    def extract_jersey_color(frame: np.ndarray, bbox: List[float]) -> np.ndarray:
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
            bgr_pixels = cv2.cvtColor(
                saturated_pixels.reshape(-1, 1, 3), cv2.COLOR_HSV2BGR
            )
            lab_pixels = cv2.cvtColor(bgr_pixels, cv2.COLOR_BGR2LAB).reshape(-1, 3)
        else:
            lab_pixels = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB).reshape(-1, 3)
        return np.median(lab_pixels, axis=0).astype(np.float32)

    def classify_video(
        self, video_path: str, video_data: List[Dict[str, Any]]
    ) -> List[Dict[int, int]]:
        """Fit both team colors together and assign labels by persistent track."""
        track_features: Dict[int, List[np.ndarray]] = {}
        frame_features: List[List[tuple[Optional[int], int, np.ndarray]]] = [
            [] for _ in video_data
        ]
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        started_at = time.perf_counter()
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
                            track_id = player.get("track_id")
                            if track_id is not None:
                                track_features.setdefault(int(track_id), []).append(
                                    feature
                                )
                            frame_features[frame_index].append(
                                (
                                    int(track_id) if track_id is not None else None,
                                    player_index,
                                    feature,
                                )
                            )
                frame_index += 1
                report_frame_progress(
                    "Jersey-color feature extraction",
                    frame_index,
                    total_frames,
                    started_at,
                )
        finally:
            cap.release()

        frame_teams = [dict() for _ in video_data]
        track_centers = {
            track_id: np.median(samples, axis=0)
            for track_id, samples in track_features.items()
        }
        cluster_tracks = list(track_centers)
        samples_to_cluster = (
            np.asarray([track_centers[track_id] for track_id in cluster_tracks])
            if len(cluster_tracks) >= self.n_clusters
            else np.asarray(
                [
                    feature
                    for frame_samples in frame_features
                    for _, _, feature in frame_samples
                ]
            )
        )
        if len(samples_to_cluster) < self.n_clusters:
            print(
                "[processing] Team classification skipped: too few jersey samples.",
                flush=True,
            )
            return frame_teams

        print(
            f"[processing] Clustering {len(samples_to_cluster)} jersey samples.",
            flush=True,
        )
        kmeans = KMeans(n_clusters=self.n_clusters, n_init=10, random_state=42)
        labels = kmeans.fit_predict(samples_to_cluster)
        if np.linalg.norm(kmeans.cluster_centers_[0] - kmeans.cluster_centers_[1]) < 8:
            print(
                "[processing] Team colors are too similar to classify confidently.",
                flush=True,
            )
            return frame_teams

        center_order = sorted(
            range(self.n_clusters),
            key=lambda index: tuple(kmeans.cluster_centers_[index].tolist()),
        )
        canonical_labels = {
            old_label: new_label for new_label, old_label in enumerate(center_order)
        }
        track_labels = {
            track_id: canonical_labels[int(label)]
            for track_id, label in zip(cluster_tracks, labels)
        }
        for frame_index, frame_samples in enumerate(frame_features):
            for track_id, player_index, feature in frame_samples:
                old_label = (
                    track_labels[track_id]
                    if track_id in track_labels
                    else canonical_labels[int(kmeans.predict(feature.reshape(1, -1))[0])]
                )
                frame_teams[frame_index][player_index] = old_label

        print(
            f"[processing] Team classification complete: "
            f"{sum(map(len, frame_teams))} detections assigned.",
            flush=True,
        )
        return frame_teams