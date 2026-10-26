import os 
import re
import asyncio
from typing import AsyncIterator
from sqlalchemy.orm import declarative_base
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from dotenv import load_dotenv

if os.name == "nt" and hasattr(asyncio, "WindowsSelectorEventLoopPolicy"):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

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

async def init_db() -> None:
  """Create application tables if they do not exist."""
  async with engine.begin() as conn:
    await conn.run_sync(Base.metadata.create_all)

async def get_db() -> AsyncIterator[AsyncSession]:
  """Dependency to get a database session."""
  async with AsyncSessionLocal() as session:
    try:
      yield session
    except Exception as e:
      await session.rollback()
      raise e
