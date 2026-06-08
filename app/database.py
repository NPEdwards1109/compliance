"""SQLAlchemy engine, session factory, and schema initialization."""
from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Generator

from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session, sessionmaker

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./data/compliance.db")

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False},
)

SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)


def _init_fts(conn) -> None:
    """Create FTS5 virtual tables and sync triggers."""
    # Documents FTS: search across title + summary + full_text
    conn.execute(text("""
        CREATE VIRTUAL TABLE IF NOT EXISTS documents_fts
        USING fts5(title, summary, full_text, content='documents', content_rowid='rowid')
    """))
    conn.execute(text("""
        CREATE TRIGGER IF NOT EXISTS documents_fts_insert
        AFTER INSERT ON documents BEGIN
            INSERT INTO documents_fts(rowid, title, summary, full_text)
            VALUES (new.rowid, new.title, new.summary, new.full_text);
        END
    """))
    conn.execute(text("""
        CREATE TRIGGER IF NOT EXISTS documents_fts_update
        AFTER UPDATE ON documents BEGIN
            INSERT INTO documents_fts(documents_fts, rowid, title, summary, full_text)
            VALUES ('delete', old.rowid, old.title, old.summary, old.full_text);
            INSERT INTO documents_fts(rowid, title, summary, full_text)
            VALUES (new.rowid, new.title, new.summary, new.full_text);
        END
    """))
    conn.execute(text("""
        CREATE TRIGGER IF NOT EXISTS documents_fts_delete
        AFTER DELETE ON documents BEGIN
            INSERT INTO documents_fts(documents_fts, rowid, title, summary, full_text)
            VALUES ('delete', old.rowid, old.title, old.summary, old.full_text);
        END
    """))

    # Requirements FTS: search requirement text
    conn.execute(text("""
        CREATE VIRTUAL TABLE IF NOT EXISTS requirements_fts
        USING fts5(text, content='requirements', content_rowid='rowid')
    """))
    conn.execute(text("""
        CREATE TRIGGER IF NOT EXISTS requirements_fts_insert
        AFTER INSERT ON requirements BEGIN
            INSERT INTO requirements_fts(rowid, text) VALUES (new.rowid, new.text);
        END
    """))
    conn.execute(text("""
        CREATE TRIGGER IF NOT EXISTS requirements_fts_update
        AFTER UPDATE ON requirements BEGIN
            INSERT INTO requirements_fts(requirements_fts, rowid, text)
            VALUES ('delete', old.rowid, old.text);
            INSERT INTO requirements_fts(rowid, text) VALUES (new.rowid, new.text);
        END
    """))
    conn.execute(text("""
        CREATE TRIGGER IF NOT EXISTS requirements_fts_delete
        AFTER DELETE ON requirements BEGIN
            INSERT INTO requirements_fts(requirements_fts, rowid, text)
            VALUES ('delete', old.rowid, old.text);
        END
    """))


def init_db() -> None:
    """Create all tables and FTS infrastructure (idempotent)."""
    from app.models import Base
    Base.metadata.create_all(engine)
    with engine.connect() as conn:
        _init_fts(conn)
        conn.commit()


@contextmanager
def get_session() -> Generator[Session, None, None]:
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
