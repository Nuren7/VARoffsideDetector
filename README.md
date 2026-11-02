# VAR Offside Detector

VAR Offside Detector is a football computer-vision project that analyses match footage and visualises player and ball detections alongside a candidate offside line. It combines YOLO object detection, jersey-colour clustering, and OpenCV video processing.

> **Note:** This is an experimental analysis tool, not an official VAR system or an authoritative offside decision.

## Features

- Detects people and sports balls with Ultralytics YOLO (COCO classes `0` and `32`); ball detection uses a lower confidence threshold and larger inference size for small balls.
- Tracks detections between frames with ByteTrack and smooths tracked boxes and candidate lines with short temporal median filters.
- Extracts bounding boxes and confidence scores; keeps the highest-confidence ball detection in each frame.
- Clusters upper-body jersey colors across the whole video and assigns team IDs per player track to reduce label flicker.
- Can map player footpoints to pitch coordinates with a per-frame homography and draw the candidate line in perspective.
- Freezes a candidate line and the attacking players' positions at a heuristic ball-release event instead of deciding again on every frame.

## Technology and deployment

| Component | Technology |
| --- | --- |
| AI and video processing | Python, Ultralytics YOLO, OpenCV, NumPy, scikit-learn |
| Frontend | React 19, Vite — deployed on Vercel |
| Backend | Python — deployed as a Northflank Web Service |
| Database | Neon PostgreSQL |

## Repository layout

```text
backend/
  app/ai/
    detector.py     # YOLO inference and detection extraction
    clustering.py   # Jersey-colour clustering
    geometry.py     # Video annotation and candidate offside line
  requirements.txt
  weights/          # YOLO model weights
frontend/           # React + Vite application
```

## Local development

### Backend

Requires Python 3.10 or newer. In PowerShell, from the repository root:

```powershell
cd backend
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Place `yolo11n.pt` in `backend/weights/`. The detector uses the relative path `weights/yolo11n.pt`, so run it from the `backend` directory.

Team 0 is configured to attack right by default, and team 1 attacks left. Set `TEAM_0_ATTACKING_DIRECTION=left` in `backend/.env` to reverse them. Team IDs are assigned in a consistent order based on the clustered jersey colors, but do not represent club names; check that team 0 is the team moving right in your clip.

### Pitch calibration

Without pitch calibration, the detector falls back to image coordinates and draws a vertical screen-space line. This fallback cannot correct camera perspective. For a fixed camera, create a JSON calibration file containing at least four matching, visible pitch landmarks:

```json
{
  "pitch_length_m": 105,
  "pitch_width_m": 68,
  "image_points": [[120, 140], [1100, 100], [1300, 650], [40, 700]],
  "pitch_points_m": [[0, 0], [105, 0], [105, 68], [0, 68]]
}
```

Replace the example image coordinates with the actual pixel positions of the corresponding pitch landmarks in your video, in the same order. Then set `PITCH_CALIBRATION_PATH=path\\to\\your-calibration.json` in `backend/.env`. A fixed homography is only suitable while the camera remains still.

For broadcast footage or a moving camera, use a YOLO pose/keypoint model trained to return pitch landmarks. Set `PITCH_MODEL_PATH=weights\\your-pitch-keypoints.pt` and provide `PITCH_KEYPOINTS_JSON` as a JSON object mapping that model's keypoint indices to pitch coordinates, for example `{"0":[0,0],"1":[105,0],"2":[105,68],"3":[0,68]}`. The model's keypoint order and coordinates must match the mapping; no pitch keypoint model is bundled with this repository. A detected homography is smoothed across frames and briefly reused when a frame has too few visible landmarks.

Optional pitch dimensions can be set with `PITCH_LENGTH_M` and `PITCH_WIDTH_M`; the defaults are 105 m by 68 m. The detected ball's lower bounding-box point is used as an approximate ground contact point, so an airborne ball can still introduce error.

### Frontend

```powershell
cd frontend
npm install
npm run dev
```

## Limitations

The output is a visual position estimate, not an offside decision. The ball-release event is heuristic, not a confirmed teammate touch; it cannot reliably distinguish deliberate opponent play, deflections, rebounds, or active involvement. A pitch homography is only as good as the supplied landmarks/model and is invalid for image regions outside the visible pitch. A person detector cannot reliably distinguish referees from players or goalkeepers from outfield players, so jersey clustering can misclassify them. Tracking can still lose or switch IDs during occlusion or crowded scenes. Player feet and the ball's lower bounding-box point approximate their ground positions; bounding boxes do not identify legal scoring body parts. Similar jersey colors, lighting, camera movement, and ball size can all reduce accuracy.

## Disclaimer

This independent project is not affiliated with FIFA, UEFA, or any official VAR organisation. Its output is for educational and experimental use only.
