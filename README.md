# South Korea RAG: document ingestion

The document-ingestion module reads supported source files from `data/` (including
subdirectories) and splits their text into semantic chunks using OpenAI
embeddings. Chunk boundaries are based on changes in meaning between adjacent
sentences; chunks are not limited to a fixed number of characters or tokens.

## Supported sources

- PDF: `.pdf`
- Word: `.docx`
- Plain text and Markdown: `.txt`, `.md`
- URLs: `.url` shortcut files containing a URL or `URL=https://...`; a `.txt`
  file containing only a URL is also fetched. HTML pages, PDF, DOCX, and plain
  text responses are supported.

## Setup

Install the project dependencies and configure the project-root `.env` file with
your OpenAI API key and database settings:

```env
OPENAI_API_KEY=your_openai_api_key_here
DATABASE_URL=postgresql://rag_user:rag_local_password@localhost:5433/south_korea_rag
POSTGRES_DB=south_korea_rag
POSTGRES_USER=rag_user
POSTGRES_PASSWORD=choose_a_local_password
POSTGRES_PORT=5433
```

Keep credentials private; `.env` is ignored by Git. Semantic chunking uses
`text-embedding-3-small` through LangChain's `OpenAIEmbeddings`.

## Usage

```python
from src.document_ingestion.document_processor import process_documents

chunks = process_documents()
```

`process_documents(data_dir=..., threshold_percentile=90.0)` can be used to
choose another input directory or adjust the semantic-distance percentile.
Smaller percentile values produce more boundaries; larger values produce fewer.
You can also call `load_documents()` and `semantic_chunk_documents()` separately.

## PostgreSQL + pgvector

The local vector database runs in Docker and persists its data in a named volume.
After configuring `.env`, start PostgreSQL:

```powershell
docker compose up -d
```

Once the database is healthy and `OPENAI_API_KEY` is set, load files from `data/`,
semantically chunk them, create embeddings, and store the chunk text and complete
metadata JSON in PostgreSQL:

```powershell
uv run python -m src.vectorstore.postgres_vector_store
```

The database schema is initialized automatically, and rerunning indexing replaces
the stored chunks for each source included in that run. Embeddings use
`text-embedding-3-small` (1536 dimensions); the table has a cosine HNSW index.
Keep the sample database password for local development only and change it for
any deployment exposed beyond the local machine. A hosted PostgreSQL provider
with the pgvector extension can be used by setting `DATABASE_URL` to its
connection string; it must support PostgreSQL 16-compatible pgvector operations.
The local Docker port defaults to `5433` to avoid conflicts with an existing
PostgreSQL server on the standard port; set `POSTGRES_PORT` to override it.

For application code:

```python
from src.vectorstore import PostgresVectorStore, index_data_directory

stored_count = index_data_directory()
store = PostgresVectorStore()
results = store.similarity_search("South Korea climate policy", k=5)
```

Each search result is a `(Document, cosine_similarity)` pair. An optional
`metadata_filter` matches JSON metadata, for example
`{"type": "pdf"}`.

## ReAct research graph

The LangGraph ReAct agent follows an agent → tool action → observation loop. It
first retrieves up to four relevant passages from the PostgreSQL document
collection and supplies them to the agent as its primary evidence. The agent
then uses Wikipedia or Tavily only when needed to supplement gaps, background,
or requested current information; external results should not replace the
project-document evidence. If project documents provide no relevant evidence,
the answer should say so and distinguish any external information.

The agent can perform additional document searches, Wikipedia lookups, and
Tavily web searches in its ReAct loop. After receiving each tool result, it can
choose another action or produce its final answer. Set `OPENAI_API_KEY`,
`DATABASE_URL`, and `TAVILY_API_KEY` in the project `.env` file. The OpenAI chat
model defaults to `gpt-4o-mini`; set `RAG_MODEL` to use another compatible model.
Start PostgreSQL and index documents as described above before using the graph.

```python
from src.graph_builder import build_rag_graph

graph = build_rag_graph()
result = graph.invoke({"question": "What are South Korea's climate policies?"})
print(result["generation"])
```

The returned state includes the final answer, conversation messages, and any
project documents retrieved during the agent's tool calls. The agent allows up
to six tool-calling iterations before asking the model for a final answer.

## FastAPI backend and Streamlit frontend

The FastAPI backend exposes `GET /health`, `POST /api/chat`, and interactive
API documentation at `/docs`. It builds the RAG graph on the first chat request.
The Streamlit frontend renders the PESTEL and scenario questions as buttons.
Selecting a question sends it and recent conversation turns to `POST /api/chat`,
then displays the generated answer, project-document sources, and the tools
actually used by the ReAct agent. The UI shows completed tool actions, not the
agent's private chain-of-thought. Failures from supplementary Wikipedia or web
searches are returned to the agent as failed observations so it can continue
using project-document evidence; core retrieval or other unexpected errors are
still reported as API errors. The UI also reports connection errors if the
backend is unavailable.

After installing dependencies and configuring `OPENAI_API_KEY`,
`DATABASE_URL`, and `TAVILY_API_KEY`, run these in separate terminals from the
project root:

```powershell
uv run uvicorn src.api.main:app --reload
```

```powershell
uv run streamlit run streamlit_app.py
```

The frontend defaults to `http://127.0.0.1:8000`. Set `RAG_API_URL` in the
frontend environment to point it at another backend URL.