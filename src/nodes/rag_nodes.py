"""ReAct agent node, routing logic, and tools for the RAG graph."""

from __future__ import annotations

import os
from typing import Annotated, Sequence

from langchain_community.tools.wikipedia.tool import WikipediaQueryRun
from langchain_community.utilities.wikipedia import WikipediaAPIWrapper
from langchain_core.documents import Document
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    AnyMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.tools import BaseTool, InjectedToolCallId, tool
from langchain_tavily import TavilySearch
from langgraph.types import Command

from src.state import RAGState
from src.vectorstore import PostgresVectorStore

MAX_TOOL_ITERATIONS = 6
RETRIEVAL_COUNT = 4
SYSTEM_PROMPT = """You are a careful research assistant focused on South Korea.
Project-document evidence is retrieved before you are called and is the
PRIMARY source for the answer. Base the main explanation and conclusions on
those passages, citing their source names when available. Evaluate whether the
passages are relevant and sufficient; do not treat unrelated passages as
evidence or invent support they do not contain.

Wikipedia and tavily_search are SECONDARY sources. Use them only when the
project documents leave a material gap, need brief background, or current
information is requested. They should support or contextualize the
document-based answer, not replace or contradict it silently. If external
information differs from the project documents, explain the difference and
identify which source says what. If project documents provide no relevant
evidence, say so explicitly and label any externally supported information as
supplementary.

Use the available tools as a ReAct loop: assess the evidence, take a useful
tool action if needed, inspect its observation, and then decide whether
another action is necessary or whether to answer. Do not claim that a search
succeeded if it returned no results. Provide a clear answer with citations to
source names or URLs when available. Do not reveal private chain-of-thought;
show only the final answer and completed tool actions."""


def _search_project_documents(
    vector_store: PostgresVectorStore,
    query: str,
) -> tuple[list[Document], str]:
    results = vector_store.similarity_search(query, k=RETRIEVAL_COUNT)
    documents = [document for document, _ in results]
    if not results:
        return documents, "No relevant project documents were found."

    context = "\n\n".join(
        (
            f"Source: {document.metadata.get('source', 'unknown')}\n"
            f"Cosine similarity: {score:.3f}\n"
            f"Passage: {document.page_content}"
        )
        for document, score in results
    )
    return documents, context


def create_primary_retrieval_node(vector_store: PostgresVectorStore):
    """Retrieve primary project-document evidence before the ReAct agent runs."""

    def retrieve_primary_documents(state: RAGState) -> dict[str, object]:
        question = state["question"].strip()
        if not question:
            raise ValueError("question must not be empty")
        documents, context = _search_project_documents(vector_store, question)
        return {
            "documents": documents,
            "retrieval_context": context,
        }

    return retrieve_primary_documents


def create_rag_tools(
    vector_store: PostgresVectorStore,
    tavily_search: BaseTool,
    wikipedia_search: BaseTool,
) -> list[BaseTool]:
    """Create the three tools available to the ReAct agent."""

    @tool("retrieve_documents")
    def retrieve_documents(
        query: str,
        tool_call_id: Annotated[str, InjectedToolCallId],
    ) -> Command:
        """Search for additional project-document passages if needed."""
        documents, content = _search_project_documents(vector_store, query)
        return Command(
            update={
                "documents": documents,
                "messages": [
                    ToolMessage(
                        content=content,
                        name="retrieve_documents",
                        tool_call_id=tool_call_id,
                    )
                ],
            }
        )

    return [retrieve_documents, wikipedia_search, tavily_search]


def create_agent_node(
    model: BaseChatModel,
    tools: Sequence[BaseTool],
):
    """Create a node that reasons, requests tools, or returns a final answer."""
    model_with_tools = model.bind_tools(tools)

    def run_agent(state: RAGState) -> dict[str, object]:
        previous_messages = state.get("messages", [])
        iteration = state.get("iterations", 0)
        if iteration == 0:
            question = state["question"].strip()
            if not question:
                raise ValueError("question must not be empty")
            human_message = HumanMessage(content=question)
            conversation = [*previous_messages, human_message]
            new_messages: list[AnyMessage] = [human_message]
        else:
            conversation = list(previous_messages)
            new_messages = []

        iteration += 1
        selected_model = (
            model_with_tools if iteration <= MAX_TOOL_ITERATIONS else model
        )
        response = selected_model.invoke(
            [
                SystemMessage(
                    content=(
                        f"{SYSTEM_PROMPT}\n\n"
                        "Primary project-document retrieval for this question:\n"
                        f"{state.get('retrieval_context', 'No retrieval context was provided.')}"
                    )
                ),
                *conversation,
            ]
        )
        if not isinstance(response, AIMessage):
            raise TypeError("The chat model must return an AIMessage")
        new_messages.append(response)
        update: dict[str, object] = {
            "messages": new_messages,
            "iterations": iteration,
        }
        if not getattr(response, "tool_calls", []):
            answer = _message_text(response)
            if not answer.strip():
                raise ValueError("The chat model returned an empty final answer")
            update["generation"] = answer
        return update

    return run_agent


def route_after_agent(state: RAGState) -> str:
    """Continue the ReAct loop for tool calls; otherwise return the answer."""
    last_message = state["messages"][-1]
    if getattr(last_message, "tool_calls", []):
        return "tools"
    return "end"


def _message_text(message: AIMessage) -> str:
    content = message.content
    if isinstance(content, str):
        return content
    return "\n".join(
        block["text"]
        for block in content
        if isinstance(block, dict) and isinstance(block.get("text"), str)
    )


def create_tavily_search_tool() -> BaseTool:
    """Create the Tavily tool from the configured environment."""
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        raise ValueError(
            "TAVILY_API_KEY is required. Set it in the environment or project .env file."
        )

    return TavilySearch(
        max_results=5,
        name="tavily_search",
        handle_tool_error=False,
    )


def create_wikipedia_search_tool() -> BaseTool:
    """Create the Wikipedia search tool."""
    return WikipediaQueryRun(
        api_wrapper=WikipediaAPIWrapper(
            top_k_results=3,
            doc_content_chars_max=4000,
        )
    )
