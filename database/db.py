"""
OrthoTwin AI — Database Connection & Engine Configuration
Provides the SQLAlchemy Base, Engine, Sessionmaker, and initialization routines.
"""

from contextlib import contextmanager
from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker
from config import DATABASE_URL, DATABASE_ECHO

# Create engine with thread-safe settings for SQLite
connect_args = {"check_same_thread": False} if "sqlite" in DATABASE_URL else {}
engine = create_engine(
    DATABASE_URL,
    echo=DATABASE_ECHO,
    connect_args=connect_args
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


def init_db():
    """Initializes database schema and ensures all tables exist."""
    import database.models  # Ensure models are loaded into Base metadata
    Base.metadata.create_all(bind=engine)


@contextmanager
def get_db():
    """Context manager for scoped database sessions."""
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()
