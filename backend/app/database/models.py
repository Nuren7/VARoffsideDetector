import datetime
from sqlalchemy import Column, Integer, String, DateTime
from app.database.connection import Base

class VideoAnalysis(Base):

    """
    VideoAnalysis model represents the analysis of a video."""
    __tablename__ = "video_analysis"

    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String, nullable=False)
    raw_video_url = Column(String, nullable=True)
    ai_video_url = Column(String, nullable=True)
    status = Column(String, default="pending")  # pending, processing, completed, failed
    created_at = Column(DateTime, default=datetime.datetime.utcnow)