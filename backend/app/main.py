import os
import shutil
from fastapi import FastAPI, UploadFile, Depends, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.ext.asyncio import AsyncSession

# Import custom modules
from app.database.connection import get_db
from app.database.models import VideoAnalysis
from app.services.storage import upload_video_to_cloudinary

# (AI modules can be integrated later when ready)
# from app.ai.detector import FootballDetector
# from app.ai.clustering import TeamClustering
# from app.ai.geometry import OffsideAnalyzer

app = FastAPI(
    title="VAR Offside Detector",
    description="A FastAPI application for detecting offside situations in football videos using AI.", 
    version="1.0.0"
)

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
    """Endpoint to upload and process a video file."""
    temp_input_path = f"temp_{file.filename}"
    temp_output_path = f"output_offside_{file.filename}"
    
    try:
        # Save uploaded file locally as temporary
        with open(temp_input_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        # Create a record in the database with pending status
        db_record = VideoAnalysis(
            filename=file.filename, 
            status="pending",
        )
        db.add(db_record)
        await db.commit()
        await db.refresh(db_record)
        
        # detector = FootballDetector()
        # raw_data = detector.process_video(temp_input_path)
        # clustering = TeamClustering()
        # player_teams = clustering.assign_teams(temp_input_path, raw_data)
        # analyzer = OffsideAnalyzer(output_video_path=temp_output_path)
        # analyzer.process_and_draw(temp_input_path, raw_data, player_teams)
        
        # AI pipeline (Simulated with a dummy file for now)
        if not os.path.exists(temp_output_path):
            shutil.copy(temp_input_path, temp_output_path)
        
        # Upload the result to cloud storage (Cloudinary)
        ai_video_url = upload_video_to_cloudinary(temp_output_path)
    
        # Update the database with the result
        db_record.ai_video_url = ai_video_url
        db_record.status = "completed"
        await db.commit()
    
        return {
            "message": "Video uploaded and processed successfully.",
            "video_analysis_id": db_record.id,
            "filename": file.filename,
            "ai_video_url": ai_video_url,
            "status": "SUCCESS",
        }
        
    except Exception as e:
        # If an error occurs, update status to Failed in the database
        if "db_record" in locals():
            db_record.status = "failed"
            await db.commit()
            
        raise HTTPException(
            status_code=500, 
            detail=f"An error occurred during processing: {e}"
        )
        
    finally:
        # Clean up temporary files from disk
        if os.path.exists(temp_input_path):
            os.remove(temp_input_path)
        if os.path.exists(temp_output_path):
            os.remove(temp_output_path)