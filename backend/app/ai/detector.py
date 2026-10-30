from typing import Any, Dict, List, Optional

import cv2
from ultralytics import YOLO


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

    def process_video(self, video_path: str) -> List[Dict[str, Any]]:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")

        video_data: List[Dict[str, Any]] = []
        inference_confidence = min(
            self.confidence_threshold, self.ball_confidence_threshold
        )

        try:
            while True:
                success, frame = cap.read()
                if not success:
                    break

                result = self.model(
                    frame,
                    conf=inference_confidence,
                    imgsz=self.image_size,
                    classes=[0, 32],
                    verbose=False,
                )[0]
                frame_players: List[Dict[str, Any]] = []
                frame_ball: Optional[Dict[str, Any]] = None

                if result.boxes is not None and len(result.boxes) > 0:
                    boxes = result.boxes.xyxy.cpu().numpy()
                    classes = result.boxes.cls.cpu().numpy()
                    confidences = result.boxes.conf.cpu().numpy()

                    for box, class_id, confidence in zip(boxes, classes, confidences):
                        class_id = int(class_id)
                        confidence = float(confidence)
                        detection = {
                            "bbox": box.tolist(),
                            "confidence": confidence,
                        }

                        if class_id == 0 and confidence >= self.confidence_threshold:
                            frame_players.append(detection)
                        elif (
                            class_id == 32
                            and confidence >= self.ball_confidence_threshold
                            and (
                                frame_ball is None
                                or confidence > frame_ball["confidence"]
                            )
                        ):
                            frame_ball = detection

                video_data.append(
                    {
                        "players": frame_players,
                        "ball": frame_ball,
                    }
                )

            return video_data
        finally:
            cap.release()
