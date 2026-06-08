"""SQLAlchemy models for the compliance knowledge graph."""
from __future__ import annotations

from datetime import datetime

from sqlalchemy import Column, DateTime, ForeignKey, Integer, PrimaryKeyConstraint, String, Text
from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    pass


class Document(Base):
    __tablename__ = "documents"

    id = Column(String, primary_key=True)  # UUID
    title = Column(String, nullable=False)
    doc_type = Column(String, nullable=False)  # regulation, standard, guidance, article, enforcement_action
    jurisdiction = Column(String)  # EU, US-Federal, US-CO, US-TX, US-CA, UK, Global, etc.
    issuer = Column(String)  # European Parliament, NIST, ISO, FTC, etc.
    status = Column(String, default="enacted")  # enacted, proposed, draft, superseded
    effective_date = Column(String)  # YYYY-MM-DD
    url = Column(String)
    full_text = Column(Text)
    summary = Column(Text)
    embedding = Column(Text)  # JSON array of floats
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class Section(Base):
    __tablename__ = "sections"

    id = Column(String, primary_key=True)  # UUID
    doc_id = Column(String, ForeignKey("documents.id"), nullable=False)
    section_number = Column(String)  # e.g. "Article 6", "Section 3.2"
    title = Column(String)
    content = Column(Text, nullable=False)
    embedding = Column(Text)  # JSON array of floats
    parent_section_id = Column(String, ForeignKey("sections.id"))


class Requirement(Base):
    __tablename__ = "requirements"

    id = Column(String, primary_key=True)  # UUID
    doc_id = Column(String, ForeignKey("documents.id"), nullable=False)
    section_id = Column(String, ForeignKey("sections.id"))
    text = Column(Text, nullable=False)
    obligation_type = Column(String)  # MUST, SHOULD, MAY
    applies_to = Column(Text)  # JSON array: ["providers", "deployers", "high-risk-systems"]
    risk_level = Column(String)  # unacceptable, high, limited, minimal, unspecified
    embedding = Column(Text)  # JSON array of floats


class DocRelationship(Base):
    __tablename__ = "doc_relationships"

    id = Column(Integer, primary_key=True, autoincrement=True)
    from_id = Column(String, ForeignKey("documents.id"), nullable=False)
    to_id = Column(String, ForeignKey("documents.id"), nullable=False)
    relationship = Column(String, nullable=False)  # CITES, AMENDS, IMPLEMENTS, SUPERSEDES, RELATED_TO, ANALYZED_BY
    notes = Column(Text)


class Topic(Base):
    __tablename__ = "topics"

    id = Column(String, primary_key=True)  # UUID
    name = Column(String, nullable=False, unique=True)
    parent_id = Column(String, ForeignKey("topics.id"))
    description = Column(Text)


class DocTopic(Base):
    __tablename__ = "doc_topics"

    doc_id = Column(String, ForeignKey("documents.id"), nullable=False)
    topic_id = Column(String, ForeignKey("topics.id"), nullable=False)
    __table_args__ = (PrimaryKeyConstraint("doc_id", "topic_id"),)
