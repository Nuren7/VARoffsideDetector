import os
import cloudinary
import cloudinary.uploader
from dotenv import load_dotenv

#get env from .env
load_dotenv()

# Configure Cloudinary with your credentials

cloudinary.config(
  cloud_name = os.getenv("CLOUDINARY_CLOUD_NAME"),
  api_key = os.getenv("CLOUDINARY_API_KEY"),
  api_secret = os.getenv("CLOUDINARY_API_SECRET"),
  secure = True
)

def upload_video_to_cloudinary(file_path: str) -> str:
  """Uploads a video file to Cloudinary and returns the URL of the uploaded video."""
  if not os.path.exists(file_path):
    raise FileNotFoundError(f"File not found: {file_path}")
  
  try:
    #load upp file video to a dedicated folder
    response = cloudinary.uploader.upload_large(
      file_path,
      resource_type="video",
      folder="offside_detector_results"
    )
    
    #return the secure URL of the uploaded video
    secure_url = response.get("secure_url")
    return secure_url 
  
  except Exception as e:
    raise RuntimeError(f"Failed to upload video to Cloudinary: {e}")
