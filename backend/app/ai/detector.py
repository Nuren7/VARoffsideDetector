import time
from typing import Any, Dict, List, Optional

import cv2
from ultralytics import YOLO

from app.ai.progress import report_frame_progress
from app.ai.pitch import PitchCalibrator


class FotballDetector:
    def __init__(
        self,
        model_path: str = "weights/yolo11n.pt",
        confidence_threshold: float = 0.3,
        ball_confidence_threshold: float = 0.08,
        image_size: int = 1280,
    ):
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.ball_confidence_threshold = ball_confidence_threshold
        self.image_size = image_size
        self.model = YOLO(model_path)
        self.ball_model = YOLO(model_path)

    def process_video(
        self,
        video_path: str,
        pitch_calibrator: Optional[PitchCalibrator] = None,
    ) -> List[Dict[str, Any]]:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")

        video_data: List[Dict[str, Any]] = []
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        started_at = time.perf_counter()
        total_label = str(total_frames) if total_frames > 0 else "unknown"
        print(
            f"[processing] YOLO detection started ({total_label} frames).",
            flush=True,
        )

        try:
            while True:
                success, frame = cap.read()
                if not success:
                    break

                result = self.model.track(
                    frame,
                    persist=True,
                    tracker="bytetrack.yaml",
                    conf=min(
                        self.confidence_threshold,
                        self.ball_confidence_threshold,
                    ),
                    imgsz=self.image_size,
                    classes=[0],
                    verbose=False,
                )[0]
                ball_result = self.ball_model(
                    frame,
                    conf=self.ball_confidence_threshold,
                    imgsz=self.image_size,
                    classes=[32],
                    verbose=False,
                )[0]
                frame_players: List[Dict[str, Any]] = []
                frame_ball: Optional[Dict[str, Any]] = None

                if result.boxes is not None and len(result.boxes) > 0:
                    boxes = result.boxes.xyxy.cpu().numpy()
                    confidences = result.boxes.conf.cpu().numpy()
                    track_ids = (
                        result.boxes.id.int().cpu().tolist()
                        if result.boxes.id is not None
                        else [None] * len(boxes)
                    )

                    for box, confidence, track_id in zip(
                        boxes, confidences, track_ids
                    ):
                        confidence = float(confidence)
                        frame_players.append(
                            {
                                "bbox": box.tolist(),
                                "confidence": confidence,
                                "track_id": track_id,
                            }
                        )

                if ball_result.boxes is not None and len(ball_result.boxes) > 0:
                    boxes = ball_result.boxes.xyxy.cpu().numpy()
                    confidences = ball_result.boxes.conf.cpu().numpy()
                    best_index = int(confidences.argmax())
                    frame_ball = {
                        "bbox": boxes[best_index].tolist(),
                        "confidence": float(confidences[best_index]),
                    }

                pitch_homography = (
                    pitch_calibrator.homography_for_frame(frame)
                    if pitch_calibrator is not None
                    else None
                )
                video_data.append(
                    {
                        "players": frame_players,
                        "ball": frame_ball,
                        "pitch_homography": (
                            pitch_homography.tolist()
                            if pitch_homography is not None
                            else None
                        ),
                    }
                )
                report_frame_progress(
                    "YOLO detection",
                    len(video_data),
                    total_frames,
                    started_at,
                )

            report_frame_progress(
                "YOLO detection",
                len(video_data),
                total_frames,
                started_at,
                force=True,
            )
            ball_frames = sum(frame["ball"] is not None for frame in video_data)
            print(
                f"[processing] YOLO detection complete: {len(video_data)} frames; "
                f"ball detected in {ball_frames} frames.",
                flush=True,
            )
            return video_data
        finally:
            cap.release()
