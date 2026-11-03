"""Build the LangGraph ReAct workflow for South Korea research."""

from __future__ import annotations

import json
import os
from pathlib import Path

import requests
from dotenv import load_dotenv
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.tools import BaseTool, ToolException
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode
from wikipedia.exceptions import WikipediaException

from src.nodes.rag_nodes import (
    create_agent_node,
    create_primary_retrieval_node,
    create_rag_tools,
    route_after_agent,
    create_tavily_search_tool,
    create_wikipedia_search_tool,
)
from src.state import RAGInput, RAGState
from src.vectorstore import PostgresVectorStore

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OPTIONAL_TOOL_ERRORS = (
    ToolException,
    requests.RequestException,
    json.JSONDecodeError,
    WikipediaException,
)


def build_rag_graph(
    *,
    model: BaseChatModel | None = None,
    vector_store: PostgresVectorStore | None = None,
    tavily_search: BaseTool | None = None,
    wikipedia_search: BaseTool | None = None,
):
    """Create a compiled ReAct graph using retrieval, Wikipedia, and Tavily.

    Dependencies may be supplied by callers for tests or custom deployments.
    The default graph reads credentials from the environment or project .env.
    """
    load_dotenv(PROJECT_ROOT / ".env")

    if model is None:
        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError(
                "OPENAI_API_KEY is required. Set it in the environment or project .env file."
            )
        model = ChatOpenAI(
            model=os.getenv("RAG_MODEL", "gpt-4o-mini"),
            temperature=0,
        )
    if vector_store is None:
        vector_store = PostgresVectorStore()
    if tavily_search is None:
        tavily_search = create_tavily_search_tool()
    if wikipedia_search is None:
        wikipedia_search = create_wikipedia_search_tool()

    tools = create_rag_tools(vector_store, tavily_search, wikipedia_search)
    graph = StateGraph(RAGState, input_schema=RAGInput)
    graph.add_node(
        "primary_retrieval",
        create_primary_retrieval_node(vector_store),
    )
    graph.add_node("agent", create_agent_node(model, tools))
    graph.add_node(
        "tools",
        ToolNode(tools, handle_tool_errors=OPTIONAL_TOOL_ERRORS),
    )
    graph.set_entry_point("primary_retrieval")
    graph.add_edge("primary_retrieval", "agent")
    graph.add_conditional_edges(
        "agent",
        route_after_agent,
        {"tools": "tools", "end": END},
    )
    graph.add_edge("tools", "agent")
    return graph.compile()
