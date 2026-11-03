"""Load source documents and split them at embedding-detected topic changes."""

from __future__ import annotations

import os
import re
from io import BytesIO
from math import sqrt
from pathlib import Path
from typing import Sequence
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup
from docx import Document as WordDocument
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_openai import OpenAIEmbeddings
from pypdf import PdfReader

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"

_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+|\n+")
_URL_PATTERN = re.compile(r"^https?://\S+$", re.IGNORECASE)
_SUPPORTED_FILE_TYPES = {".pdf", ".docx", ".txt", ".md", ".url"}


def load_documents(data_dir: str | Path = DATA_DIR) -> list[Document]:
    """Load PDFs, Word documents, text/Markdown files, and URL shortcut files.

    URL shortcut files use the ``.url`` extension and contain a URL (or a
    Windows InternetShortcut ``URL=...`` entry). A text file containing only
    one URL is also treated as a URL source.
    """
    load_dotenv(PROJECT_ROOT / ".env")
    root = Path(data_dir).expanduser()
    if not root.exists():
        raise FileNotFoundError(f"Document directory does not exist: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"Document path is not a directory: {root}")

    documents: list[Document] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in _SUPPORTED_FILE_TYPES:
            continue
        try:
            if path.suffix.lower() == ".url":
                documents.append(_load_url_file(path))
            elif path.suffix.lower() == ".txt":
                text = path.read_text(encoding="utf-8-sig")
                url = _extract_url(text)
                documents.append(_load_url(url) if url else _local_text_document(path, text))
            elif path.suffix.lower() == ".md":
                documents.append(_local_text_document(path, path.read_text(encoding="utf-8-sig")))
            else:
                documents.append(_load_local_file(path))
        except Exception as exc:
            raise RuntimeError(f"Failed to load document {path}: {exc}") from exc
    return [document for document in documents if document.page_content.strip()]


def semantic_chunk_documents(
    documents: Sequence[Document],
    embeddings: Embeddings | None = None,
    threshold_percentile: float = 90.0,
) -> list[Document]:
    """Split documents where consecutive sentence embeddings change meaning.

    The percentile controls how large a semantic distance must be to create a
    boundary. It is not a chunk-size limit; resulting chunks vary in length.
    """
    if not 0 <= threshold_percentile <= 100:
        raise ValueError("threshold_percentile must be between 0 and 100")

    nonempty_documents = [doc for doc in documents if doc.page_content.strip()]
    if not nonempty_documents:
        return []

    load_dotenv(PROJECT_ROOT / ".env")
    embedding_model = embeddings
    chunks: list[Document] = []

    for document in nonempty_documents:
        sentences = _split_sentences(document.page_content)
        if len(sentences) <= 1:
            chunks.append(_make_chunk(document, sentences, 0))
            continue

        if embedding_model is None:
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError(
                    "OPENAI_API_KEY is required for semantic chunking. "
                    "Set it in the environment or project .env file."
                )
            embedding_model = OpenAIEmbeddings(
                model="text-embedding-3-small",
                api_key=api_key,
            )
        vectors = embedding_model.embed_documents([sentence for sentence, _ in sentences])
        distances = [
            1.0 - _cosine_similarity(left, right)
            for left, right in zip(vectors, vectors[1:])
        ]
        threshold = _percentile(distances, threshold_percentile)

        current_start = 0
        chunk_index = 0
        for boundary_index, distance in enumerate(distances):
            if distance > threshold:
                chunks.append(
                    _make_chunk(document, sentences[current_start : boundary_index + 1], chunk_index)
                )
                current_start = boundary_index + 1
                chunk_index += 1
        chunks.append(_make_chunk(document, sentences[current_start:], chunk_index))

    return chunks


def process_documents(
    data_dir: str | Path = DATA_DIR,
    embeddings: Embeddings | None = None,
    threshold_percentile: float = 90.0,
) -> list[Document]:
    """Load files from ``data_dir`` and return semantically chunked documents."""
    documents = load_documents(data_dir)
    return semantic_chunk_documents(
        documents,
        embeddings=embeddings,
        threshold_percentile=threshold_percentile,
    )


def _load_url_file(path: Path) -> Document:
    content = path.read_text(encoding="utf-8-sig")
    url = _extract_url(content)
    if url is None:
        match = re.search(r"(?im)^\s*URL\s*=\s*(\S+)\s*$", content)
        url = match.group(1) if match else None
    if url is None:
        raise ValueError("URL shortcut must contain a valid http:// or https:// URL")
    return _load_url(url)


def _extract_url(content: str) -> str | None:
    candidate = content.strip()
    return candidate if _URL_PATTERN.fullmatch(candidate) else None


def _load_url(url: str) -> Document:
    parsed = urlparse(url)
    if parsed.scheme.lower() not in {"http", "https"} or not parsed.netloc:
        raise ValueError(f"Only absolute http:// and https:// URLs are supported: {url}")

    response = requests.get(
        url,
        timeout=30,
        headers={"User-Agent": "SouthKoreaRAG/1.0"},
    )
    response.raise_for_status()
    content_type = response.headers.get("Content-Type", "").lower()
    suffix = Path(parsed.path).suffix.lower()
    source = response.url

    if suffix == ".pdf" or "application/pdf" in content_type:
        text = _pdf_text(BytesIO(response.content))
    elif suffix == ".docx" or "wordprocessingml" in content_type:
        text = _word_text(BytesIO(response.content))
    elif "text/html" in content_type or not content_type:
        soup = BeautifulSoup(response.content, "html.parser")
        title = soup.title.get_text(" ", strip=True) if soup.title else ""
        for element in soup(["script", "style", "noscript", "svg"]):
            element.decompose()
        text = soup.get_text("\n", strip=True)
        return Document(
            page_content=text,
            metadata={"source": source, "title": title, "type": "url"},
        )
    elif content_type.startswith("text/"):
        text = response.text
    else:
        raise ValueError(f"Unsupported URL content type: {content_type or 'unknown'}")

    return Document(page_content=text, metadata={"source": source, "type": "url"})


def _load_local_file(path: Path) -> Document:
    if path.suffix.lower() == ".pdf":
        text = _pdf_text(path)
        kind = "pdf"
    else:
        text = _word_text(path)
        kind = "docx"
    return Document(
        page_content=text,
        metadata={"source": str(path.resolve()), "type": kind},
    )


def _pdf_text(source: Path | BytesIO) -> str:
    reader = PdfReader(source)
    return "\n\n".join(page.extract_text() or "" for page in reader.pages)


def _word_text(source: Path | BytesIO) -> str:
    document = WordDocument(source)
    parts = [paragraph.text for paragraph in document.paragraphs if paragraph.text.strip()]
    for table in document.tables:
        for row in table.rows:
            cells = [cell.text.strip() for cell in row.cells]
            if any(cells):
                parts.append(" | ".join(cells))
    return "\n".join(parts)


def _local_text_document(path: Path, text: str) -> Document:
    return Document(
        page_content=text,
        metadata={"source": str(path.resolve()), "type": path.suffix.lower().lstrip(".")},
    )


def _split_sentences(text: str) -> list[tuple[str, int]]:
    sentences: list[tuple[str, int]] = []
    cursor = 0
    for match in _SENTENCE_BOUNDARY.finditer(text):
        raw_sentence = text[cursor : match.start()]
        sentence = raw_sentence.strip()
        if sentence:
            start = cursor + len(raw_sentence) - len(raw_sentence.lstrip())
            sentences.append((sentence, start))
        cursor = match.end()

    raw_sentence = text[cursor:]
    sentence = raw_sentence.strip()
    if sentence:
        start = cursor + len(raw_sentence) - len(raw_sentence.lstrip())
        sentences.append((sentence, start))
    return sentences


def _cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right):
        raise ValueError("Embedding vectors must have matching dimensions")
    left_norm = sqrt(sum(value * value for value in left))
    right_norm = sqrt(sum(value * value for value in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return sum(a * b for a, b in zip(left, right)) / (left_norm * right_norm)


def _percentile(values: Sequence[float], percentile: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile / 100
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def _make_chunk(
    document: Document,
    sentences: Sequence[tuple[str, int]],
    chunk_index: int,
) -> Document:
    text = " ".join(sentence for sentence, _ in sentences)
    metadata = dict(document.metadata)
    metadata.update(
        {
            "chunk_index": chunk_index,
            "start_index": sentences[0][1] if sentences else 0,
        }
    )
    return Document(page_content=text, metadata=metadata)
