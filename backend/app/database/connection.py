import os 
import re
from typing import AsyncIterator
from sqlalchemy.orm import declarative_base
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from dotenv import load_dotenv

#load env from .env
load_dotenv()

#get database url from neon
DATABASE_URL = os.getenv("DATABASE_URL")

if not DATABASE_URL:
  raise ValueError("Database URL not found. Please set the DATABASE_URL environment variable.")

# Use the async psycopg driver, which is installed by requirements.txt.
async_database_url = re.sub(
  r"^postgres(?:ql)?://", "postgresql+psycopg://", DATABASE_URL
)

# Create the SQLAlchemy engine
engine = create_async_engine(async_database_url, echo=True)

# Create a sessionmaker for creating database sessions
AsyncSessionLocal = async_sessionmaker(bind=engine, class_=AsyncSession, expire_on_commit=False)

# Create a base class for declarative models
Base = declarative_base()

async def get_db() -> AsyncIterator[AsyncSession]:
  """Dependency to get a database session."""
  async with AsyncSessionLocal() as session:
    try:
      yield session
    except Exception as e:
      await session.rollback()
      raise e
