"""Shared state schemas for the retrieval-augmented generation graph."""

from operator import add
from typing import Annotated, NotRequired, TypedDict

from langchain_core.documents import Document
from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages


class RAGState(TypedDict):
    """Data passed between the ReAct agent and its tools."""

    question: str
    messages: Annotated[list[AnyMessage], add_messages]
    documents: Annotated[list[Document], add]
    retrieval_context: NotRequired[str]
    generation: NotRequired[str]
    iterations: NotRequired[int]


class RAGInput(TypedDict):
    """Input accepted when invoking the compiled graph."""

    question: str
    messages: NotRequired[Annotated[list[AnyMessage], add_messages]]
