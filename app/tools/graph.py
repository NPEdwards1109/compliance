"""Graph traversal tools: navigate relationships between documents."""
from __future__ import annotations

from typing import Optional

from fastmcp import FastMCP

from app.database import get_session
from app.models import DocRelationship, DocTopic, Document, Topic

_VALID_RELATIONSHIPS = {"CITES", "AMENDS", "IMPLEMENTS", "SUPERSEDES", "RELATED_TO", "ANALYZED_BY"}


def register_graph_tools(mcp: FastMCP) -> None:

    @mcp.tool()
    def get_related(doc_id: str, relationship: Optional[str] = None) -> dict:
        """Find documents related to a given document via the knowledge graph.

        Returns both outbound (this doc relates to others) and inbound (others relate to this doc).
        Filter by relationship type: CITES, AMENDS, IMPLEMENTS, SUPERSEDES, RELATED_TO, ANALYZED_BY.
        """
        with get_session() as session:
            doc = session.get(Document, doc_id)
            if not doc:
                return {"error": f"Document {doc_id!r} not found"}

            outbound_q = session.query(DocRelationship).filter_by(from_id=doc_id)
            inbound_q = session.query(DocRelationship).filter_by(to_id=doc_id)
            if relationship:
                rel_upper = relationship.upper()
                outbound_q = outbound_q.filter(DocRelationship.relationship == rel_upper)
                inbound_q = inbound_q.filter(DocRelationship.relationship == rel_upper)

            outbound = outbound_q.all()
            inbound = inbound_q.all()

            def _doc_stub(did: str) -> dict:
                d = session.get(Document, did)
                if not d:
                    return {"id": did, "title": "unknown"}
                return {
                    "id": d.id,
                    "title": d.title,
                    "doc_type": d.doc_type,
                    "jurisdiction": d.jurisdiction,
                    "status": d.status,
                }

            return {
                "doc_id": doc_id,
                "title": doc.title,
                "outbound": [
                    {
                        "relationship": r.relationship,
                        "notes": r.notes,
                        "document": _doc_stub(r.to_id),
                    }
                    for r in outbound
                ],
                "inbound": [
                    {
                        "relationship": r.relationship,
                        "notes": r.notes,
                        "document": _doc_stub(r.from_id),
                    }
                    for r in inbound
                ],
            }

    @mcp.tool()
    def link_documents(
        from_id: str,
        to_id: str,
        relationship: str,
        notes: Optional[str] = None,
    ) -> dict:
        """Create a directed relationship edge between two documents.

        relationship must be one of: CITES, AMENDS, IMPLEMENTS, SUPERSEDES, RELATED_TO, ANALYZED_BY

        Examples:
        - EU AI Act IMPLEMENTS ISO 42001
        - NIST AI RMF RELATED_TO EU AI Act
        - Colorado AI Act CITES EU AI Act
        """
        rel_upper = relationship.upper()
        if rel_upper not in _VALID_RELATIONSHIPS:
            return {"error": f"Invalid relationship. Must be one of: {sorted(_VALID_RELATIONSHIPS)}"}

        with get_session() as session:
            from_doc = session.get(Document, from_id)
            to_doc = session.get(Document, to_id)
            if not from_doc:
                return {"error": f"Document {from_id!r} not found"}
            if not to_doc:
                return {"error": f"Document {to_id!r} not found"}

            # Check for duplicate
            existing = (
                session.query(DocRelationship)
                .filter_by(from_id=from_id, to_id=to_id, relationship=rel_upper)
                .first()
            )
            if existing:
                return {"status": "already_exists", "relationship": rel_upper}

            edge = DocRelationship(
                from_id=from_id,
                to_id=to_id,
                relationship=rel_upper,
                notes=notes,
            )
            session.add(edge)

            return {
                "status": "created",
                "from": {"id": from_doc.id, "title": from_doc.title},
                "relationship": rel_upper,
                "to": {"id": to_doc.id, "title": to_doc.title},
            }

    @mcp.tool()
    def get_topic_coverage(topic_name: str) -> list[dict]:
        """Find which documents cover a given topic.

        Topics include high-level themes like 'high-risk AI', 'transparency',
        'human oversight', 'data governance', 'conformity assessment', etc.
        """
        with get_session() as session:
            # Search topic by name (partial match)
            topics = session.query(Topic).filter(
                Topic.name.ilike(f"%{topic_name}%")
            ).all()

            if not topics:
                return [{"message": f"No topics found matching {topic_name!r}. Use list_topics to see available topics."}]

            results = []
            for topic in topics:
                doc_topics = session.query(DocTopic).filter_by(topic_id=topic.id).all()
                docs = []
                for dt in doc_topics:
                    doc = session.get(Document, dt.doc_id)
                    if doc:
                        docs.append({
                            "id": doc.id,
                            "title": doc.title,
                            "jurisdiction": doc.jurisdiction,
                            "doc_type": doc.doc_type,
                            "status": doc.status,
                        })
                results.append({
                    "topic": topic.name,
                    "description": topic.description,
                    "document_count": len(docs),
                    "documents": docs,
                })

            return results

    @mcp.tool()
    def list_topics() -> list[dict]:
        """List all topics in the knowledge taxonomy."""
        with get_session() as session:
            topics = session.query(Topic).order_by(Topic.name).all()
            return [
                {
                    "id": t.id,
                    "name": t.name,
                    "parent_id": t.parent_id,
                    "description": t.description,
                }
                for t in topics
            ]

    @mcp.tool()
    def get_graph(include_mermaid: bool = True) -> dict:
        """Return the full knowledge graph: all documents as nodes and all relationships as edges.

        With include_mermaid=True (default), also returns a Mermaid diagram string
        that can be rendered visually to understand the graph structure at a glance.

        Useful for understanding how regulations relate to each other — what implements what,
        what supersedes what, what cites what.
        """
        with get_session() as session:
            docs = session.query(Document).order_by(Document.effective_date).all()
            edges = session.query(DocRelationship).all()

            nodes = [
                {
                    "id": d.id,
                    "title": d.title,
                    "doc_type": d.doc_type,
                    "jurisdiction": d.jurisdiction,
                    "status": d.status,
                    "effective_date": d.effective_date,
                    "issuer": d.issuer,
                }
                for d in docs
            ]

            edge_list = [
                {
                    "from_id": e.from_id,
                    "from_title": next((d.title for d in docs if d.id == e.from_id), e.from_id[:8]),
                    "relationship": e.relationship,
                    "to_id": e.to_id,
                    "to_title": next((d.title for d in docs if d.id == e.to_id), e.to_id[:8]),
                    "notes": e.notes,
                }
                for e in edges
            ]

            result: dict = {"node_count": len(nodes), "edge_count": len(edge_list),
                            "nodes": nodes, "edges": edge_list}

            if include_mermaid:
                # Build short labels: acronym or first ~30 chars
                def _label(doc: Document) -> str:
                    acronyms = {
                        "EU Artificial Intelligence Act": "EU AI Act",
                        "NIST AI Risk Management Framework (AI RMF 1.0)": "NIST AI RMF",
                        "NIST AI 600-1 — Generative AI Profile": "NIST AI 600-1",
                        "ISO/IEC 42001:2023 — AI Management Systems": "ISO 42001",
                        "Executive Order 14110 — Safe, Secure, and Trustworthy AI (Biden)": "EO 14110",
                        "Executive Order 14179 — Removing Barriers to American Leadership in AI (Trump)": "EO 14179",
                        "Colorado AI Act (SB 24-205) — Consumer Protections for AI": "CO AI Act",
                        "FTC Policy Statement on AI and Consumer Protection": "FTC AI Policy",
                    }
                    return acronyms.get(doc.title, doc.title[:28])

                id_to_safe = {d.id: d.title.replace(" ", "_").replace("/", "_").replace("-", "_")[:20] + f"_{i}"
                               for i, d in enumerate(docs)}
                id_to_label = {d.id: _label(d) for d in docs}

                rel_arrow = {
                    "SUPERSEDES": "-->|supersedes|",
                    "IMPLEMENTS": "-->|implements|",
                    "CITES": "-->|cites|",
                    "AMENDS": "-->|amends|",
                    "RELATED_TO": "<-->|related|",
                    "ANALYZED_BY": "-->|analyzed by|",
                }

                lines = ["graph TD"]
                for d in docs:
                    safe = id_to_safe[d.id]
                    label = id_to_label[d.id]
                    jurisdiction_tag = f" [{d.jurisdiction}]"
                    lines.append(f'    {safe}["{label}{jurisdiction_tag}"]')
                lines.append("")
                seen_related: set[frozenset] = set()
                for e in edges:
                    if e.from_id not in id_to_safe or e.to_id not in id_to_safe:
                        continue
                    # Deduplicate bidirectional RELATED_TO edges
                    if e.relationship == "RELATED_TO":
                        pair = frozenset([e.from_id, e.to_id])
                        if pair in seen_related:
                            continue
                        seen_related.add(pair)
                    arrow = rel_arrow.get(e.relationship, f"-->|{e.relationship.lower()}|")
                    lines.append(f"    {id_to_safe[e.from_id]} {arrow} {id_to_safe[e.to_id]}")

                result["mermaid"] = "\n".join(lines)

            return result

    @mcp.tool()
    def tag_document_topic(doc_id: str, topic_name: str, create_if_missing: bool = True) -> dict:
        """Associate a document with a topic.

        If the topic doesn't exist and create_if_missing=True, it's created automatically.
        """
        import uuid

        with get_session() as session:
            doc = session.get(Document, doc_id)
            if not doc:
                return {"error": f"Document {doc_id!r} not found"}

            topic = session.query(Topic).filter_by(name=topic_name).first()
            if not topic:
                if not create_if_missing:
                    return {"error": f"Topic {topic_name!r} not found"}
                topic = Topic(id=str(uuid.uuid4()), name=topic_name)
                session.add(topic)
                session.flush()

            existing = session.query(DocTopic).filter_by(doc_id=doc_id, topic_id=topic.id).first()
            if existing:
                return {"status": "already_tagged", "doc": doc.title, "topic": topic.name}

            dt = DocTopic(doc_id=doc_id, topic_id=topic.id)
            session.add(dt)

            return {"status": "tagged", "doc": doc.title, "topic": topic.name}
