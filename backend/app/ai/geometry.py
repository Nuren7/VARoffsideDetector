import cv2
import numpy as np
from typing import Any, Dict, List, Optional


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


class OffsideAnalyzer:
    """Draw a ball-relative candidate offside line and candidate attackers."""

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
        cap = cv2.VideoCapture(input_video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {input_video_path}")

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        if video_fps <= 0 or np.isnan(video_fps):
            video_fps = self.fps

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(self.output_video_path, fourcc, video_fps, (width, height))
        if not out.isOpened():
            out.release()
            fourcc = cv2.VideoWriter_fourcc(*"avc1")
            out = cv2.VideoWriter(
                self.output_video_path, fourcc, video_fps, (width, height)
            )
        if not out.isOpened():
            cap.release()
            raise RuntimeError(
                f"Could not create output video: {self.output_video_path}"
            )

        frame_index = 0
        try:
            while True:
                success, frame = cap.read()
                if not success:
                    break

                if frame_index < len(video_data):
                    frame_data = video_data[frame_index]
                    players = frame_data.get("players", [])
                    ball = frame_data.get("ball")
                    teams = (
                        player_teams[frame_index]
                        if frame_index < len(player_teams)
                        else {}
                    )

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

                    attacking_team = self._team_nearest_ball(players, teams, ball)
                    line_x = None
                    if attacking_team is not None:
                        line_x = calculate_offside_line(
                            players,
                            teams,
                            ball,
                            1 - attacking_team,
                            self.attacking_direction,
                        )

                    if ball is not None:
                        bx1, by1, bx2, by2 = map(int, ball["bbox"])
                        center = ((bx1 + bx2) // 2, (by1 + by2) // 2)
                        cv2.rectangle(frame, (bx1, by1), (bx2, by2), (0, 255, 0), 2)
                        cv2.circle(frame, center, 5, (0, 255, 0), -1)

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
        finally:
            cap.release()
            out.release()
