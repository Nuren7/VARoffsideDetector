import os 
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base
from dotenv import load_dotenv

#load env from .env
load_dotenv()

#get database url from neon
DATABASE_URL = os.getenv("DATABASE_URL")

if not DATABASE_URL:
  raise ValueError("Database URL not found. Please set the DATABASE_URL environment variable.")

# Create the SQLAlchemy engine
engine = create_engine(DATABASE_URL)

#create session local to handle temporary sessions
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Create a base class for declarative models
Base = declarative_base()

def get_db():
  """Dependency to get a database session."""
  db = SessionLocal()
  try:
    yield db
  finally:
    db.close()
    
