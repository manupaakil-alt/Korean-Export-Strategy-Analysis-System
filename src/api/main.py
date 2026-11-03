"""FastAPI endpoints for the RAG assistant."""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import Literal

from fastapi import Depends, FastAPI, HTTPException
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.graph.state import CompiledStateGraph
from pydantic import BaseModel, Field, field_validator

from src.graph_builder import build_rag_graph

logger = logging.getLogger(__name__)

app = FastAPI(
    title="South Korea RAG API",
    description="Research assistant using indexed documents, Wikipedia, and Tavily.",
    version="0.1.0",
)


class ChatHistoryMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=8000)


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    history: list[ChatHistoryMessage] = Field(default_factory=list, max_length=20)

    @field_validator("question")
    @classmethod
    def question_must_not_be_blank(cls, question: str) -> str:
        question = question.strip()
        if not question:
            raise ValueError("question must not be blank")
        return question


class Source(BaseModel):
    source: str
    type: str | None = None


class ToolStep(BaseModel):
    tool: str
    status: Literal["completed", "failed"] = "completed"


class ChatResponse(BaseModel):
    answer: str
    sources: list[Source]
    tool_steps: list[ToolStep]


@lru_cache(maxsize=1)
def _build_graph() -> CompiledStateGraph:
    return build_rag_graph()


def get_rag_graph() -> CompiledStateGraph:
    """Resolve the shared graph, reporting initialization errors as unavailable."""
    try:
        return _build_graph()
    except Exception as exc:
        logger.exception("Could not initialize the RAG graph")
        raise HTTPException(
            status_code=503,
            detail="RAG service is not configured. Check backend logs and credentials.",
        ) from exc


@app.get("/health")
def health() -> dict[str, str]:
    """Report that the API process is running."""
    return {"status": "ok"}


@app.post("/api/chat", response_model=ChatResponse)
def chat(
    request: ChatRequest,
    graph: CompiledStateGraph = Depends(get_rag_graph),
) -> ChatResponse:
    """Answer a question, optionally using recent user/assistant chat history."""
    history = [
        HumanMessage(content=message.content)
        if message.role == "user"
        else AIMessage(content=message.content)
        for message in request.history
    ]
    try:
        result = graph.invoke(
            {
                "question": request.question,
                "messages": history,
            }
        )
    except Exception as exc:
        logger.exception("RAG graph invocation failed")
        raise HTTPException(
            status_code=502,
            detail="The RAG request failed. Check backend logs for details.",
        ) from exc

    answer = result.get("generation")
    if not isinstance(answer, str) or not answer.strip():
        logger.error("RAG graph completed without a final answer")
        raise HTTPException(
            status_code=502,
            detail="The RAG service did not return an answer.",
        )

    sources_by_name: dict[str, Source] = {}
    for document in result.get("documents", []):
        source_name = document.metadata.get("source")
        if not isinstance(source_name, str) or not source_name:
            continue
        source_type = document.metadata.get("type")
        sources_by_name[source_name] = Source(
            source=source_name,
            type=source_type if isinstance(source_type, str) else None,
        )

    tool_steps = [
        ToolStep(tool="primary_document_retrieval"),
        *[
            ToolStep(
                tool=message.name,
                status="failed" if message.status == "error" else "completed",
            )
            for message in result.get("messages", [])
            if isinstance(message, ToolMessage)
            and isinstance(message.name, str)
            and message.name
        ],
    ]

    return ChatResponse(
        answer=answer,
        sources=list(sources_by_name.values()),
        tool_steps=tool_steps,
    )
