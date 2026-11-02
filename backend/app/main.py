import json
import os
import shutil
from fastapi import FastAPI, UploadFile, Depends, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.ext.asyncio import AsyncSession

# Import custom modules
from app.database.connection import get_db, init_db
from app.database.models import VideoAnalysis
from app.services.storage import upload_video_to_cloudinary

# Import AI modules
from app.ai.detector import FotballDetector
from app.ai.clustering import Teamclassifier
from app.ai.geometry import OffsideAnalyzer
from app.ai.pitch import PitchCalibrator

app = FastAPI(
    title="VAR Offside Detector",
    description="A FastAPI application for detecting offside situations in football videos using AI.", 
    version="1.0.0"
)

@app.on_event("startup")
async def startup_event():
    await init_db()

# Configure CORS to allow the frontend to access the backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this to your frontend's origin in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    """Root endpoint to check if the API is running."""
    return {"status": "online", "message": "Welcome to the VAR Offside Detector API!"}

@app.post("/api/upload")
async def upload_and_process_video(
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_db)
):
    """Endpoint to upload and process a video file using the AI pipeline."""
    temp_input_path = f"temp_{file.filename}"
    temp_output_path = f"output_offside_{file.filename}"
    
    try:
        print(f"[processing] Upload received: {file.filename}", flush=True)
        # 1. Save uploaded file locally as temporary
        with open(temp_input_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        # 2. Create a record in the database with pending status
        db_record = VideoAnalysis(
            filename=file.filename, 
            status="pending",
        )
        db.add(db_record)
        await db.commit()
        await db.refresh(db_record)
        
        # 3. AI Pipeline Step 1: Detect players and ball with YOLO
        print("[processing] Loading YOLO model.", flush=True)
        pitch_calibrator = None
        calibration_path = os.getenv("PITCH_CALIBRATION_PATH")
        pitch_model_path = os.getenv("PITCH_MODEL_PATH")
        pitch_length_m = float(os.getenv("PITCH_LENGTH_M", "105"))
        pitch_width_m = float(os.getenv("PITCH_WIDTH_M", "68"))
        if calibration_path or pitch_model_path:
            keypoint_mapping = json.loads(
                os.getenv("PITCH_KEYPOINTS_JSON", "{}")
            )
            pitch_calibrator = PitchCalibrator(
                calibration_path=calibration_path,
                model_path=pitch_model_path,
                keypoint_pitch_points={
                    int(index): point for index, point in keypoint_mapping.items()
                },
                pitch_length_m=pitch_length_m,
                pitch_width_m=pitch_width_m,
            )
            pitch_length_m = pitch_calibrator.pitch_length_m
            pitch_width_m = pitch_calibrator.pitch_width_m
            print("[processing] Pitch calibration enabled.", flush=True)
        else:
            print(
                "[processing] Pitch calibration not configured; "
                "using image-space fallback.",
                flush=True,
            )
        detector = FotballDetector()
        raw_data = detector.process_video(temp_input_path, pitch_calibrator)
        
        # 4. AI Pipeline Step 2: Fit team colors together for stable IDs.
        team_classifier = Teamclassifier(n_clusters=2)
        player_teams = team_classifier.classify_video(temp_input_path, raw_data)
        
        # 5. AI Pipeline Step 3: Draw offside lines and generate final video
        team_zero_direction = os.getenv(
            "TEAM_0_ATTACKING_DIRECTION",
            os.getenv("ATTACKING_DIRECTION", "right"),
        ).lower()
        team_directions = {
            0: team_zero_direction,
            1: "left" if team_zero_direction == "right" else "right",
        }
        analyzer = OffsideAnalyzer(
            output_video_path=temp_output_path,
            team_directions=team_directions,
            pitch_length_m=pitch_length_m,
            pitch_width_m=pitch_width_m,
        )
        analyzer.process_and_draw(temp_input_path, raw_data, player_teams)
        
        # Fallback om filen mot förmodan inte skapades
        if not os.path.exists(temp_output_path):
            shutil.copy(temp_input_path, temp_output_path)
        
        # 6. Upload the result to cloud storage (Cloudinary)
        print("[processing] Uploading annotated video to Cloudinary.", flush=True)
        ai_video_url = upload_video_to_cloudinary(temp_output_path)
        print("[processing] Cloudinary upload complete.", flush=True)
    
        # 7. Update the database with the result and completed status
        db_record.ai_video_url = ai_video_url
        db_record.status = "completed"
        await db.commit()
        print(f"[processing] Analysis complete: {file.filename}", flush=True)
    
        return {
            "message": "Video uploaded and processed successfully with AI.",
            "video_analysis_id": db_record.id,
            "filename": file.filename,
            "ai_video_url": ai_video_url,
            "status": "SUCCESS",
        }
        
    except Exception as e:
        print(f"[processing] Analysis failed for {file.filename}: {e}", flush=True)
        # If an error occurs, update status to Failed in the database
        if "db_record" in locals():
            try:
                await db.rollback()
                db_record.status = "failed"
                await db.commit()
            except Exception:
                await db.rollback()

        raise HTTPException(
            status_code=500, 
            detail=f"An error occurred during AI processing: {e}"
        )
        
    finally:
        # Clean up temporary files from disk
        if os.path.exists(temp_input_path):
            os.remove(temp_input_path)
        if os.path.exists(temp_output_path):
            os.remove(temp_output_path)