"""Persist and search document chunks in PostgreSQL with the pgvector extension."""

from __future__ import annotations

import os
from collections import defaultdict
from pathlib import Path
from typing import Sequence
from uuid import NAMESPACE_URL, uuid5

import psycopg
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_openai import OpenAIEmbeddings
from pgvector import Vector
from pgvector.psycopg import register_vector
from psycopg.types.json import Jsonb

from src.document_ingestion.document_processor import DATA_DIR, process_documents

EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIMENSIONS = 1536
TABLE_NAME = "document_chunks"
PROJECT_ROOT = Path(__file__).resolve().parents[2]


class PostgresVectorStore:
    """Store chunk text, its embedding, and its complete metadata in pgvector."""

    def __init__(
        self,
        connection_string: str | None = None,
        embeddings: Embeddings | None = None,
    ) -> None:
        load_dotenv(PROJECT_ROOT / ".env")
        self.connection_string = connection_string or os.getenv("DATABASE_URL")
        if not self.connection_string:
            raise ValueError(
                "DATABASE_URL is required. Set it in the environment or project .env file."
            )
        self.embeddings = embeddings or self._openai_embeddings()

    @staticmethod
    def _openai_embeddings() -> OpenAIEmbeddings:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY is required to create and search document embeddings."
            )
        return OpenAIEmbeddings(model=EMBEDDING_MODEL, api_key=api_key)

    def initialize(self) -> None:
        """Create the pgvector extension, chunk table, and cosine-search index."""
        with psycopg.connect(self.connection_string, connect_timeout=10) as connection:
            connection.execute("CREATE EXTENSION IF NOT EXISTS vector")
            register_vector(connection)
            connection.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {TABLE_NAME} (
                    id UUID PRIMARY KEY,
                    source TEXT NOT NULL,
                    chunk_index INTEGER NOT NULL,
                    content TEXT NOT NULL,
                    embedding VECTOR({EMBEDDING_DIMENSIONS}) NOT NULL,
                    metadata JSONB NOT NULL,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                )
                """
            )
            connection.execute(
                f"""
                CREATE INDEX IF NOT EXISTS {TABLE_NAME}_embedding_cosine_idx
                ON {TABLE_NAME}
                USING hnsw (embedding vector_cosine_ops)
                """
            )
            connection.execute(
                f"CREATE INDEX IF NOT EXISTS {TABLE_NAME}_metadata_idx "
                f"ON {TABLE_NAME} USING gin (metadata)"
            )

    def add_documents(self, documents: Sequence[Document]) -> int:
        """Replace stored chunks for each included source; return chunks written.

        All replacement deletes and inserts run in one transaction. If embedding
        or persistence fails, PostgreSQL rolls back and retains the prior data.
        """
        if not documents:
            return 0

        sources = [str(document.metadata.get("source", "unknown")) for document in documents]
        vectors = self.embeddings.embed_documents(
            [document.page_content for document in documents]
        )
        if len(vectors) != len(documents):
            raise ValueError(
                f"Embedding service returned {len(vectors)} vectors "
                f"for {len(documents)} document chunks"
            )

        occurrences: defaultdict[tuple[str, int, str], int] = defaultdict(int)
        rows: list[tuple[object, ...]] = []
        for document, source, vector in zip(documents, sources, vectors):
            if len(vector) != EMBEDDING_DIMENSIONS:
                raise ValueError(
                    f"Expected {EMBEDDING_DIMENSIONS}-dimensional embeddings from "
                    f"{EMBEDDING_MODEL}; received {len(vector)} dimensions"
                )
            chunk_index = int(document.metadata.get("chunk_index", 0))
            identity = (source, chunk_index, document.page_content)
            occurrence = occurrences[identity]
            occurrences[identity] += 1
            row_id = uuid5(
                NAMESPACE_URL,
                f"{source}\0{chunk_index}\0{document.page_content}\0{occurrence}",
            )
            rows.append(
                (
                    row_id,
                    source,
                    chunk_index,
                    document.page_content,
                    Vector(vector),
                    Jsonb(document.metadata),
                )
            )

        with psycopg.connect(self.connection_string, connect_timeout=10) as connection:
            register_vector(connection)
            connection.execute(
                f"DELETE FROM {TABLE_NAME} WHERE source = ANY(%s)",
                (list(dict.fromkeys(sources)),),
            )
            with connection.cursor() as cursor:
                cursor.executemany(
                    f"""
                    INSERT INTO {TABLE_NAME}
                        (id, source, chunk_index, content, embedding, metadata)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    """,
                    rows,
                )
        return len(rows)

    def similarity_search(
        self,
        query: str,
        k: int = 4,
        metadata_filter: dict[str, object] | None = None,
    ) -> list[tuple[Document, float]]:
        """Return the nearest chunks and cosine similarity scores."""
        if not query.strip():
            raise ValueError("query must not be empty")
        if k < 1:
            raise ValueError("k must be at least 1")

        query_vector = self.embeddings.embed_query(query)
        if len(query_vector) != EMBEDDING_DIMENSIONS:
            raise ValueError(
                f"Expected {EMBEDDING_DIMENSIONS}-dimensional embeddings from "
                f"{EMBEDDING_MODEL}; received {len(query_vector)} dimensions"
            )
        filter_clause = ""
        parameters: list[object] = [Vector(query_vector)]
        if metadata_filter is not None:
            filter_clause = "WHERE metadata @> %s"
            parameters.append(Jsonb(metadata_filter))
        parameters.extend((Vector(query_vector), k))

        with psycopg.connect(self.connection_string, connect_timeout=10) as connection:
            register_vector(connection)
            rows = connection.execute(
                f"""
                SELECT content, metadata, 1 - (embedding <=> %s) AS similarity
                FROM {TABLE_NAME}
                {filter_clause}
                ORDER BY embedding <=> %s
                LIMIT %s
                """,
                parameters,
            ).fetchall()

        return [
            (
                Document(page_content=content, metadata=metadata),
                float(similarity),
            )
            for content, metadata, similarity in rows
        ]


def index_data_directory(
    data_dir: str | Path = DATA_DIR,
    threshold_percentile: float = 90.0,
) -> int:
    """Load, semantically chunk, and persist the documents found in ``data_dir``."""
    store = PostgresVectorStore()
    store.initialize()
    chunks = process_documents(
        data_dir,
        embeddings=store.embeddings,
        threshold_percentile=threshold_percentile,
    )
    return store.add_documents(chunks)


if __name__ == "__main__":
    count = index_data_directory()
    print(f"Indexed {count} semantic chunks into PostgreSQL.")
