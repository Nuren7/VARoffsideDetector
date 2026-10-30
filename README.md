# VAR Offside Detector

VAR Offside Detector is a football computer-vision project that analyses match footage and visualises player and ball detections alongside a candidate offside line. It combines YOLO object detection, jersey-colour clustering, and OpenCV video processing.

> **Note:** This is an experimental analysis tool, not an official VAR system or an authoritative offside decision.

## Features

- Detects people and sports balls with Ultralytics YOLO (COCO classes `0` and `32`); ball detection uses a lower confidence threshold and larger inference size for small balls.
- Extracts bounding boxes and confidence scores; keeps the highest-confidence ball detection in each frame.
- Clusters upper-body jersey colors across the whole video so team IDs remain consistent between players and frames.
- Annotates player/team boxes, detected balls, the ball-relative second-last-opponent line, and players in a candidate offside position.

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
Set `ATTACKING_DIRECTION=left` in `backend/.env` when the attacking team moves toward the left side of the video; the default is `right`.

### Frontend

```powershell
cd frontend
npm install
npm run dev
```

## Limitations

The output is a visual position estimate, not an offside decision. Team IDs are color clusters and do not identify club names. The attacking team is estimated from the player nearest the detected ball; if the ball is not detected or two defenders cannot be identified, no candidate line is drawn for that frame. The halfway line is approximated by the image center, and player bounding boxes do not distinguish arms from legal scoring body parts. The system does not yet track players or identify the exact teammate touch, deliberate opponent plays, rebounds, or active involvement. Camera perspective, occlusion, similar jersey colors, lighting, and ball size can all reduce accuracy; use calibrated footage and a ball/player model trained for the footage for stronger results.

## Disclaimer

This independent project is not affiliated with FIFA, UEFA, or any official VAR organisation. Its output is for educational and experimental use only.
