from ultralytics import YOLO
import cv2
import numpy as np
from typing import List, Dict, Optional, Any


class FotballDetector:
    def __init__(
        self, model_path="weights/yolo11n.pt", confidence_threshold: float = 0.3
    ):
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.model = YOLO(model_path)

    def process_video(self, video_path):
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")

        video_data: List[Dict[str, Any]] = []

        try:
            while True:
                success, frame = cap.read()

                if not success:
                    break

                results = self.model(
                    frame, conf=self.confidence_threshold, verbose=False
                )[0]

                frame_players: List[Dict[str, Any]] = []
                frame_ball: Optional[Dict[str, Any]] = None

                if results.boxes is not None and len(results.boxes) > 0:
                    boxes = results.boxes.xyxy.cpu().numpy()
                    classes = results.boxes.cls.cpu().numpy()
                    confs = results.boxes.conf.cpu().numpy()

                    for box, cls_id, conf in zip(boxes, classes, confs):
                        class_id = int(cls_id)
                        confidence = float(conf)
                        coords = box.tolist()

                        if class_id == 0:
                            frame_players.append(
                                {
                                    "bbox": coords,
                                    "confidence": confidence,
                                }
                            )
                        elif class_id == 32:
                            if (
                                frame_ball is None
                                or confidence > frame_ball["confidence"]
                            ):
                                frame_ball = {
                                    "bbox": coords,
                                    "confidence": confidence,
                                }

                video_data.append(
                    {
                        "players": frame_players,
                        "ball": frame_ball,
                    }
                )

                print(
                    f"Analyzed frame {len(video_data)}: {len(frame_players)} players detected, ball detected: {frame_ball is not None}"
                )

            return video_data

        finally:
            cap.release()
