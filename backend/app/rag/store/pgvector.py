"""pgvector store for the deployed instance (Supabase Postgres).

Same interface as the local store. Cosine distance via the `<=>` operator on an
ivfflat index; retrieval_score is reported as 1 - distance so higher is better,
matching the local store.
"""
from __future__ import annotations

import json

from app.rag.store.base import VectorStore
from app.rag.types import Chunk, Hit


class PgVectorStore(VectorStore):
    def __init__(self, database_url: str, dim: int):
        import psycopg
        from pgvector.psycopg import register_vector

        if not database_url:
            raise RuntimeError("DATABASE_URL is not set but vector_store=pgvector.")
        self.dim = dim
        self._conn = psycopg.connect(database_url, autocommit=True)
        self._conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        register_vector(self._conn)
        self._ensure_table()

    def _ensure_table(self) -> None:
        self._conn.execute(
            f"""
            CREATE TABLE IF NOT EXISTS chunks (
                id        TEXT PRIMARY KEY,
                doc_id    TEXT,
                source    TEXT,
                title     TEXT,
                url       TEXT,
                ordinal   INT,
                text      TEXT,
                embedding VECTOR({self.dim})
            )
            """
        )
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS chunks_embedding_idx "
            "ON chunks USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100)"
        )

    def reset(self) -> None:
        self._conn.execute("TRUNCATE chunks")

    def add(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        import numpy as np

        with self._conn.cursor() as cur:
            cur.executemany(
                """
                INSERT INTO chunks (id, doc_id, source, title, url, ordinal, text, embedding)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (id) DO UPDATE SET
                    text = EXCLUDED.text, embedding = EXCLUDED.embedding
                """,
                [
                    (c.id, c.doc_id, c.source, c.title, c.url, c.ordinal, c.text,
                     np.array(e, dtype=np.float32))
                    for c, e in zip(chunks, embeddings)
                ],
            )

    def search(self, query_embedding: list[float], k: int) -> list[Hit]:
        import numpy as np

        q = np.array(query_embedding, dtype=np.float32)
        rows = self._conn.execute(
            """
            SELECT id, doc_id, source, title, url, ordinal, text,
                   1 - (embedding <=> %s) AS score
            FROM chunks
            ORDER BY embedding <=> %s
            LIMIT %s
            """,
            (q, q, k),
        ).fetchall()
        hits: list[Hit] = []
        for r in rows:
            chunk = Chunk(id=r[0], doc_id=r[1], source=r[2], title=r[3],
                          url=r[4], ordinal=r[5], text=r[6])
            hits.append(Hit(chunk=chunk, retrieval_score=float(r[7])))
        return hits

    def count(self) -> int:
        return self._conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
