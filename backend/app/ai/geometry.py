import time
import cv2
import numpy as np
from typing import Any, Dict, List, Optional

from app.ai.progress import report_frame_progress
from app.ai.pitch import PitchCalibrator


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
        if "pitch_xy" in player:
            defender_edges.append(float(player["pitch_xy"][0]))
        else:
            x1, _, x2, _ = player["bbox"]
            defender_edges.append(x2 if attacking_direction == "right" else x1)

    if len(defender_edges) < 2:
        return None

    defender_edges.sort(reverse=attacking_direction == "right")
    second_last_defender = defender_edges[1]
    if "pitch_xy" in ball:
        ball_x = float(ball["pitch_xy"][0])
    else:
        bx1, _, bx2, _ = ball["bbox"]
        ball_x = (bx1 + bx2) / 2
    return (
        max(second_last_defender, ball_x)
        if attacking_direction == "right"
        else min(second_last_defender, ball_x)
    )


def is_offside_position(
    position: Any,
    line_x: float,
    midfield_x: float,
    attacking_direction: str,
) -> bool:
    """Test position in pitch coordinates, or use a legacy image-space bbox."""
    if isinstance(position, (int, float, np.number)):
        player_x = float(position)
    elif isinstance(position, dict) and "pitch_xy" in position:
        player_x = float(position["pitch_xy"][0])
    else:
        bbox = position["bbox"] if isinstance(position, dict) else position
        player_x = bbox[2] if attacking_direction == "right" else bbox[0]

    if attacking_direction == "right":
        return player_x > midfield_x and player_x > line_x
    if attacking_direction == "left":
        return player_x < midfield_x and player_x < line_x
    raise ValueError("attacking_direction must be 'left' or 'right'.")


def project_ground_positions(
    video_data: List[Dict[str, Any]],
    homographies: List[Optional[List[List[float]]]],
) -> List[Dict[str, Any]]:
    """Add pitch coordinates for player footpoints and ball ground-contact points."""
    projected = [
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

    for frame_index, frame_data in enumerate(projected):
        if frame_index >= len(homographies) or homographies[frame_index] is None:
            continue
        homography = np.asarray(homographies[frame_index], dtype=np.float64)
        image_points = []
        targets = []

        for player in frame_data["players"]:
            x1, _, x2, y2 = player["bbox"]
            image_points.append(((x1 + x2) / 2, y2))
            targets.append(player)

        ball = frame_data.get("ball")
        if ball is not None:
            bx1, _, bx2, by2 = ball["bbox"]
            image_points.append(((bx1 + bx2) / 2, by2))
            targets.append(ball)

        if image_points:
            pitch_points = cv2.perspectiveTransform(
                np.asarray(image_points, dtype=np.float32).reshape(-1, 1, 2),
                homography,
            ).reshape(-1, 2)
            for target, pitch_xy in zip(targets, pitch_points):
                if np.isfinite(pitch_xy).all():
                    target["pitch_xy"] = pitch_xy.tolist()

    return projected


def smooth_video_detections(
    video_data: List[Dict[str, Any]], radius: int = 2
) -> List[Dict[str, Any]]:
    """Reduce detection jitter with a temporal median for each tracked player."""
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

    observations: Dict[
        int, Dict[int, tuple[List[float], Optional[List[float]]]]
    ] = {}
    ball_observations: Dict[int, Dict[int, List[float]]] = {}
    for frame_index, frame_data in enumerate(video_data):
        for player in frame_data.get("players", []):
            track_id = player.get("track_id")
            if track_id is not None:
                observations.setdefault(int(track_id), {})[frame_index] = (
                    player["bbox"],
                    player.get("pitch_xy"),
                )
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
            nearby_pitch_points = [
                pitch_xy
                for observed_frame, (_, pitch_xy) in track_observations.items()
                if pitch_xy is not None
                and abs(observed_frame - frame_index) <= radius
            ]
            if nearby_pitch_points:
                player["pitch_xy"] = np.median(
                    nearby_pitch_points, axis=0
                ).tolist()
            elif track_observations:
                nearby_boxes = [
                    box
                    for observed_frame, (box, _) in track_observations.items()
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
    team_ids: List[Optional[int]], radius: int = 2
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
    line_positions: List[Optional[float]], radius: int = 2, max_gap: int = 2
) -> List[Optional[float]]:
    """Median-filter candidate lines and interpolate only very short gaps."""
    smoothed = list(line_positions)
    valid_indices = [
        index for index, value in enumerate(smoothed) if value is not None
    ]

    for left_index, right_index in zip(valid_indices, valid_indices[1:]):
        gap = right_index - left_index - 1
        if 0 < gap <= max_gap:
            for index in range(left_index + 1, right_index):
                ratio = (index - left_index) / (right_index - left_index)
                smoothed[index] = (
                    line_positions[left_index] * (1 - ratio)
                    + line_positions[right_index] * ratio
                )

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
        smoothed[frame_index] = float(np.median(nearby))
    return smoothed


def freeze_offside_snapshots(
    video_data: List[Dict[str, Any]],
    player_teams: List[Dict[int, int]],
    team_directions: Dict[int, str],
    pitch_length_m: float,
    image_midfield_x: float,
    use_pitch_coordinates: bool,
) -> List[Dict[str, Any]]:
    """Keep candidate line and player positions from the last detected ball release."""
    snapshots: List[Dict[str, Any]] = []
    current_snapshot: Dict[str, Any] = {
        "attacking_team": None,
        "line_x": None,
        "offside_tokens": set(),
    }
    controller: Optional[tuple[Optional[int], int, int]] = None
    previous_ball_point: Optional[np.ndarray] = None
    previous_ball_frame: Optional[int] = None

    def player_point(player: Dict[str, Any]) -> np.ndarray:
        if "pitch_xy" in player:
            return np.asarray(player["pitch_xy"], dtype=np.float64)
        x1, _, x2, y2 = player["bbox"]
        return np.asarray(((x1 + x2) / 2, y2), dtype=np.float64)

    def ball_point(ball: Dict[str, Any]) -> np.ndarray:
        if "pitch_xy" in ball:
            return np.asarray(ball["pitch_xy"], dtype=np.float64)
        x1, y1, x2, y2 = ball["bbox"]
        return np.asarray(((x1 + x2) / 2, (y1 + y2) / 2), dtype=np.float64)

    def take_snapshot(
        frame_index: int,
        attacking_team: int,
        players: List[Dict[str, Any]],
        teams: Dict[int, int],
        ball: Dict[str, Any],
    ) -> None:
        direction = team_directions[attacking_team]
        line_x = calculate_offside_line(
            players, teams, ball, 1 - attacking_team, direction
        )
        if line_x is None:
            return

        offside_tokens = set()
        midfield_x = (
            pitch_length_m / 2
            if any("pitch_xy" in player for player in players)
            else image_midfield_x
        )
        for player_index, player in enumerate(players):
            if teams.get(player_index) != attacking_team:
                continue
            if is_offside_position(
                player if "pitch_xy" in player else player["bbox"],
                line_x,
                midfield_x,
                direction,
            ):
                track_id = player.get("track_id")
                offside_tokens.add(
                    ("track", int(track_id))
                    if track_id is not None
                    else ("index", frame_index, player_index)
                )
        current_snapshot.update(
            attacking_team=attacking_team,
            line_x=line_x,
            offside_tokens=offside_tokens,
        )

    for frame_index, frame_data in enumerate(video_data):
        players = frame_data.get("players", [])
        ball = frame_data.get("ball")
        if use_pitch_coordinates and frame_data.get("pitch_homography") is None:
            snapshots.append(
                {
                    "attacking_team": current_snapshot["attacking_team"],
                    "line_x": current_snapshot["line_x"],
                    "offside_tokens": set(current_snapshot["offside_tokens"]),
                }
            )
            continue
        teams = (
            player_teams[frame_index]
            if frame_index < len(player_teams)
            else {}
        )
        ball_xy = ball_point(ball) if ball is not None else None
        nearest = None
        if ball_xy is not None:
            candidates = [
                (float(np.linalg.norm(player_point(player) - ball_xy)), index, player)
                for index, player in enumerate(players)
                if teams.get(index) in {0, 1}
            ]
            if candidates:
                nearest = min(candidates, key=lambda candidate: candidate[0])

        if controller is not None and ball_xy is not None:
            controller_track, controller_index, controller_team = controller
            current_controller = next(
                (
                    player
                    for player in players
                    if controller_track is not None
                    and player.get("track_id") == controller_track
                ),
                None,
            )
            if current_controller is None and controller_track is None:
                if controller_index < len(players):
                    current_controller = players[controller_index]

            if current_controller is not None:
                distance_from_controller = float(
                    np.linalg.norm(player_point(current_controller) - ball_xy)
                )
                if "pitch_xy" in current_controller and "pitch_xy" in ball:
                    release_distance = 2.5
                    movement_threshold = 0.45
                    control_distance = 1.8
                else:
                    player_height = max(
                        1.0,
                        float(
                            current_controller["bbox"][3]
                            - current_controller["bbox"][1]
                        ),
                    )
                    release_distance = player_height * 0.8
                    movement_threshold = player_height * 0.12
                    control_distance = player_height * 0.2
                movement = (
                    float(np.linalg.norm(ball_xy - previous_ball_point))
                    if previous_ball_point is not None
                    else 0.0
                )
                frame_gap = (
                    frame_index - previous_ball_frame
                    if previous_ball_frame is not None
                    else 1
                )
                if (
                    distance_from_controller > release_distance
                    and frame_gap <= 3
                    and movement
                    > movement_threshold * max(1, frame_gap) * 0.6
                ):
                    take_snapshot(
                        frame_index,
                        controller_team,
                        players,
                        teams,
                        ball,
                    )
                    controller = None

        if (
            controller is None
            and nearest is not None
            and ball_xy is not None
        ):
            distance, player_index, player = nearest
            if "pitch_xy" in player and "pitch_xy" in ball:
                control_distance = 1.8
            else:
                control_distance = max(
                    12.0,
                    float(player["bbox"][3] - player["bbox"][1]) * 0.2,
                )
            if distance <= control_distance:
                controller = (
                    player.get("track_id"),
                    player_index,
                    teams[player_index],
                )

        if (
            current_snapshot["line_x"] is None
            and nearest is not None
            and ball is not None
        ):
            take_snapshot(
                frame_index,
                teams[nearest[1]],
                players,
                teams,
                ball,
            )

        snapshots.append(
            {
                "attacking_team": current_snapshot["attacking_team"],
                "line_x": current_snapshot["line_x"],
                "offside_tokens": set(current_snapshot["offside_tokens"]),
            }
        )
        if ball_xy is not None:
            previous_ball_point = ball_xy
            previous_ball_frame = frame_index

    return snapshots


class OffsideAnalyzer:
    """Draw a ball-relative candidate offside line and candidate attackers."""

    def __init__(
        self,
        output_video_path: str = "output_offside.mp4",
        fps: float = 30.0,
        attacking_direction: str = "right",
        team_directions: Optional[Dict[int, str]] = None,
        pitch_length_m: float = 105.0,
        pitch_width_m: float = 68.0,
    ):
        if attacking_direction not in {"left", "right"}:
            raise ValueError("attacking_direction must be 'left' or 'right'.")
        self.output_video_path = output_video_path
        self.fps = fps
        self.attacking_direction = attacking_direction
        self.team_directions = team_directions or {
            0: attacking_direction,
            1: "left" if attacking_direction == "right" else "right",
        }
        if any(direction not in {"left", "right"} for direction in self.team_directions.values()):
            raise ValueError("Each team direction must be 'left' or 'right'.")
        self.pitch_length_m = pitch_length_m
        self.pitch_width_m = pitch_width_m

    @staticmethod
    def _team_nearest_ball(
        players: List[Dict[str, Any]],
        player_teams: Dict[int, int],
        ball: Optional[Dict[str, Any]],
    ) -> Optional[int]:
        """Determine which team is closest to the ball in the current frame."""
        if ball is None:
            return None

        if "pitch_xy" in ball:
            ball_center = ball["pitch_xy"]
        else:
            bx1, by1, bx2, by2 = ball["bbox"]
            ball_center = ((bx1 + bx2) / 2, (by1 + by2) / 2)
        nearest_distance = float("inf")
        nearest_team = None

        for player_index, player in enumerate(players):
            team_id = player_teams.get(player_index)
            if team_id not in {0, 1}:
                continue
            if "pitch_xy" in player:
                foot_center = player["pitch_xy"]
            else:
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
        """Process the input video, compute offside lines, and draw annotations frame by frame."""
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

        homographies = [
            frame_data.get("pitch_homography") for frame_data in video_data
        ]
        has_pitch_calibration = any(homography is not None for homography in homographies)
        stable_video_data = smooth_video_detections(
            project_ground_positions(video_data, homographies)
        )
        if has_pitch_calibration:
            print(
                "[processing] Using pitch coordinates from per-frame homographies.",
                flush=True,
            )
        else:
            print(
                "[processing] No pitch homography available; using image-space "
                "fallback for this clip.",
                flush=True,
            )
        offside_snapshots = freeze_offside_snapshots(
            stable_video_data,
            player_teams,
            self.team_directions,
            self.pitch_length_m,
            width / 2,
            has_pitch_calibration,
        )

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

                    snapshot = offside_snapshots[frame_index]
                    attacking_team = snapshot["attacking_team"]
                    line_x = snapshot["line_x"]

                    # Draw the ball marker if present
                    if ball is not None:
                        bx1, by1, bx2, by2 = map(int, ball["bbox"])
                        center = ((bx1 + bx2) // 2, (by1 + by2) // 2)
                        cv2.rectangle(frame, (bx1, by1), (bx2, by2), (0, 255, 0), 2)
                        cv2.circle(frame, center, 5, (0, 255, 0), -1)

                    # Draw candidate offside line and check for offside positions
                    if line_x is not None:
                        current_line_x = int(round(line_x))
                        homography = homographies[frame_index]
                        if homography is not None:
                            PitchCalibrator.draw_pitch_line(
                                frame,
                                line_x,
                                self.pitch_width_m,
                                np.asarray(homography, dtype=np.float64),
                            )
                        elif not has_pitch_calibration:
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
                            track_id = player.get("track_id")
                            offside_token = (
                                ("track", int(track_id))
                                if track_id is not None
                                else ("index", frame_index, player_index)
                            )
                            if offside_token in snapshot["offside_tokens"]:
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