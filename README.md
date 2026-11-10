# SentinelAI --- Agentic RAG with Hybrid Retrieval and Observability

SentinelAI is an agentic Retrieval-Augmented Generation (RAG)
application that answers questions using internal documents and, when
needed, external information sources. It combines hybrid retrieval,
evidence assessment, tool-calling, and LLM-based answer generation in a
stateful LangGraph workflow.

The application supports PDF, Markdown, and text documents. It uses
ChromaDB for persistent vector storage, BM25 for keyword retrieval,
Reciprocal Rank Fusion (RRF) to combine retrieval rankings, and a
CrossEncoder to rerank candidate passages. Google Gemini is used for
query planning, evidence assessment, research/tool selection, and answer
generation.

SentinelAI also includes a FastAPI backend, a Streamlit interface,
LangSmith tracing, custom request and retrieval monitoring, and
evaluation scripts for RAG quality and agent tool selection.

## Key Features

### 1. Document ingestion and chunking

-   Supports `.pdf`, `.md`, and `.txt` documents.
-   Extracts text and splits it using LangChain's
    `RecursiveCharacterTextSplitter`.
-   Preserves source metadata and chunk positions.
-   Uses overlapping chunks to retain context across chunk boundaries.
-   Validates missing and empty documents.

Current configuration:

``` python
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
```

These values are character-based, not token-based.

### 2. Embeddings and persistent vector storage

-   Uses `sentence-transformers/all-MiniLM-L6-v2` through
    `HuggingFaceEmbeddings`.
-   Generates embeddings locally.
-   Stores document text, embeddings, metadata, and IDs in ChromaDB.
-   Uses the persistent collection `sentinel_documents`.
-   Generates deterministic chunk IDs using source information, content
    hashes, and chunk indexes.
-   Supports re-indexing and removal of stale chunks when a source
    document changes.
-   Supports document deletion from the vector store.

### 3. Hybrid retrieval

Combines two retrieval methods: - **Dense retrieval:** Uses embedding
similarity and ChromaDB to find semantically related passages. -
**Sparse retrieval:** Uses BM25 to find lexical and keyword matches.

The BM25 retriever is built from the indexed document corpus and cached
in memory. A corpus fingerprint is used to determine whether the cached
retriever needs to be refreshed.

### 4. Reciprocal Rank Fusion

Dense and BM25 retrieval produce rankings with different score scales.
SentinelAI combines their rankings using Reciprocal Rank Fusion (RRF),
which uses ranking positions rather than adding raw retrieval scores.

The current RRF constant is `60`.

\[ RRF(d)=`\sum`{=tex}\_{r
`\in `{=tex}R}`\frac{1}{k_0+\operatorname{rank}_r(d)}`{=tex} \]

### 5. CrossEncoder reranking

A CrossEncoder scores query-document pairs and reorders candidate
passages by relevance.

Model: `cross-encoder/ms-marco-MiniLM-L-6-v2`

Current configuration:

``` python
DENSE_K = 10
SPARSE_K = 10
RERANK_TOP_K = 5
RRF_CONSTANT = 60
```

These are current configuration values, not claims of empirically
optimal settings.

### 6. Agentic workflow with LangGraph

LangGraph orchestrates question answering using shared state, nodes, and
conditional routing.

The workflow includes: - **Query planner:** Uses Gemini structured
output to decide whether the question needs internal documents, web
search, Wikipedia, or calculation. - **Document retrieval:** Runs hybrid
retrieval and CrossEncoder reranking when internal documents are
required. - **Evidence evaluator:** Assesses whether retrieved passages
sufficiently cover the question, partially cover it, or are
insufficient. - **Document answer generation:** Answers directly from
internal evidence when it is sufficient and no external source was
requested. - **Research agent:** Uses Gemini tool calling to decide
whether external research or calculation is needed. - **Tool
execution:** Uses LangGraph's `ToolNode` to execute requested external
tools and return results to the research agent. - **Finalization:**
Produces a final response when the research iteration limit is reached.

The research agent has access to these external tools:

  -----------------------------------------------------------------------
  Tool                                Purpose
  ----------------------------------- -----------------------------------
  `web_search`                        Searches the web using DDGS and
                                      returns result titles, URLs, and
                                      snippets.

  `wikipedia_search`                  Searches Wikipedia and returns page
                                      summaries and URLs.

  `calculator`                        Evaluates supported mathematical
                                      expressions using a restricted
                                      AST-based evaluator.
  -----------------------------------------------------------------------

Internal document retrieval is implemented as a dedicated LangGraph node
in the current graph. The `document_rag` tool helper also exists in
`app/agent_tools.py`, but it is not bound to the current research agent.

The graph supports iterative tool execution, subject to its configured
iteration and recursion limits.

### 7. Answer and source handling

-   Generates answers using Google Gemini through LangChain.
-   Uses retrieved internal passages as context for document-grounded
    answers.
-   Passes internal evidence and evidence-assessment results to the
    research agent when external research is required.
-   Extracts and deduplicates source information from internal document
    results and external tool messages.
-   Returns the question, answer, and sources from `answer_question()`.

Retrieved document text and external search results are treated as
information to evaluate, not as instructions that should override the
system's behaviour.

### 8. LangSmith tracing

LangSmith is used to inspect supported LLM and workflow executions.

The application loads LangSmith configuration from environment variables
and uses `@traceable` instrumentation on important workflow and
retrieval functions. Instrumented operations include query planning,
document retrieval, evidence evaluation, answer generation, research,
finalization, dense retrieval, BM25 retrieval, RRF, and reranking.

Tracing helps inspect supported run inputs and outputs, execution
structure, latency, and failures. Exact trace coverage depends on the
LangChain integrations and explicit instrumentation.

### 9. Custom monitoring

`app/monitoring.py` provides lightweight in-process monitoring.

Request-level metrics include: - Total requests. - Successful
requests. - Failed requests. - Total and average request latency. -
Average number of returned sources. - Error counts by exception type. -
A bounded history of recent requests.

Stage-level metrics record execution counts, cumulative duration,
average duration, and the latest supplied details for instrumented
stages such as hybrid retrieval and reranking.

The monitoring module uses `time.perf_counter()` for elapsed-time
measurements and a thread lock to protect shared in-process metric
updates.

Metrics are held in memory. They reset when the process restarts and are
not shared across multiple worker processes.

### 10. FastAPI backend

The FastAPI application is defined in `app/api.py`.

  -------------------------------------------------------------------------
  Endpoint                  Method                  Description
  ------------------------- ----------------------- -----------------------
  `/health`                 GET                     Returns application
                                                    health information.

  `/metrics`                GET                     Returns custom
                                                    monitoring metrics.

  `/upload`                 POST                    Accepts supported
                                                    documents, enforces the
                                                    upload size limit,
                                                    saves the file, and
                                                    indexes it.

  `/documents`              GET                     Lists available
                                                    documents.

  `/documents/{filename}`   DELETE                  Deletes a document and
                                                    its indexed chunks.

  `/ask`                    POST                    Runs the LangGraph
                                                    question-answering
                                                    workflow.
  -------------------------------------------------------------------------

The `/ask` endpoint validates the incoming question and returns the
question, answer, and sources. The upload endpoint streams the incoming
file and applies a 10 MB size limit.

The API initializes the document index and BM25 cache during application
startup.

### 11. Streamlit interface

The Streamlit application is defined in `app/streamlit_app.py`. It
communicates with the FastAPI backend over HTTP and provides: - A
conversational question-answering interface. - Chat history and new-chat
handling. - Document upload. - Document listing and deletion. - Answer
and source display.

The UI uses the API rather than directly invoking the vector store or
model.

### 12. Evaluation

SentinelAI includes separate evaluation code for RAG answer quality and
agent tool-selection behaviour.

#### RAGAS evaluation

`evaluation/evaluate.py` runs questions from the evaluation dataset
through the application, collects generated answers and retrieved
contexts, and evaluates the resulting dataset using configured RAGAS
metrics.

The current evaluation script configures context precision and context
recall. These metrics assess the relevance of retrieved passages and
coverage of reference evidence, respectively.

Evaluation results depend on the dataset, model, configuration, and
evaluation run. The small current dataset is suitable for development
checks, not broad statistical claims about system quality.

#### Agent evaluation

`evaluation/agent_evaluator.py` evaluates agent behaviour against the
agentic RAG test dataset.

It checks expected and forbidden tool usage and reports
tool-selection-related measures such as required-tool recall,
tool-selection accuracy, forbidden-tool usage, and execution failures.

The current tool-use evaluation is based on the recorded plan and tool
messages. Document retrieval is inferred from the planner's document
requirement, while external tool use is identified from tool messages.
It is therefore not a complete independent audit of every graph-node
execution.

### 13. Tests

The repository includes tests and standalone checks for API behaviour,
retrieval, ChromaDB, and vector-store functionality.

Run the pytest suite from the project root:

``` powershell
python -m pytest -v
```

## Architecture

``` mermaid
flowchart TD
    A["User"] --> B["Streamlit UI"]
    B --> C["FastAPI"]
    C --> D["answer_question()"]
    D --> E["LangGraph Planner"]

    E --> F{"Needs internal documents?"}
    F -->|Yes| G["Hybrid Document Retrieval"]
    F -->|No| N["Research Agent"]

    G --> H["ChromaDB Dense Search"]
    G --> I["BM25 Sparse Search"]
    H --> J["RRF Fusion"]
    I --> J
    J --> K["CrossEncoder Reranking"]
    K --> L["Evidence Evaluator"]

    L --> M{"Sufficient and no external source requested?"}
    M -->|Yes| O["Document Answer Generation"]
    M -->|No| N

    N --> P{"External tool call?"}
    P -->|Yes| Q["LangGraph ToolNode"]
    Q --> N
    P -->|No| R["Final Answer"]

    O --> S["Extract Answer and Sources"]
    R --> S
    S --> T["FastAPI Response"]

    D -. "Request and stage metrics" .-> U["Custom Monitoring"]
    D -. "Traces" .-> V["LangSmith"]
```

LangSmith tracing and custom monitoring serve different purposes:
tracing helps inspect individual executions, while custom monitoring
aggregates request outcomes and stage timings.

## Technology Stack

  -----------------------------------------------------------------------
  Component               Technology              Purpose
  ----------------------- ----------------------- -----------------------
  Language                Python                  Application logic

  LLM                     Google Gemini           Planning, evidence
                                                  assessment, tool
                                                  selection, and answer
                                                  generation

  LLM integration         LangChain               Model, document,
                                                  retriever, and tool
                                                  abstractions

  Workflow orchestration  LangGraph               Stateful graph,
                                                  conditional routing,
                                                  and tool loop

  Embeddings              Sentence Transformers   Local semantic
                                                  embeddings

  Embedding integration   HuggingFaceEmbeddings   Loads and invokes the
                                                  embedding model

  Vector database         ChromaDB                Persistent vector
                                                  storage

  Sparse retrieval        BM25                    Keyword-based retrieval

  Ranking fusion          Reciprocal Rank Fusion  Combines dense and
                                                  sparse rankings

  Reranking               CrossEncoder            Reranks query-document
                                                  candidates

  Web search              DDGS                    External web search

  Background knowledge    Wikipedia Python        Wikipedia search and
                          library                 summaries

  Calculation             Python AST and operator Restricted mathematical
                                                  expression evaluation

  Observability           LangSmith               LLM and workflow
                                                  tracing

  Custom monitoring       Python                  Request and stage-level
                                                  metrics

  API                     FastAPI                 Backend HTTP endpoints

  UI                      Streamlit               User-facing application

  RAG evaluation          RAGAS                   Retrieval-context
                                                  evaluation

  Agent evaluation        Custom evaluator        Checks expected and
                                                  forbidden tool usage

  Testing                 pytest                  Automated tests
  -----------------------------------------------------------------------

## Project Structure

``` text
sentinalAI/
├── app/
│   ├── __init__.py
│   ├── agent_tools.py
│   ├── api.py
│   ├── graph.py
│   ├── ingestion.py
│   ├── monitoring.py
│   ├── planning.py
│   ├── retrieval.py
│   ├── streamlit_app.py
│   └── vector_store.py
│
├── data/
│   └── sample_docs/
│       └── ...
│
├── evaluation/
│   ├── agent_evaluator.py
│   ├── agentic_rag_dataset.json
│   ├── dataset.json
│   └── evaluate.py
│
├── tests/
│   └── ...
│
├── chroma_db/
├── .env.example
├── .gitignore
├── requirements.txt
└── README.md
```

The exact contents of `data/sample_docs/`, `tests/`, and local runtime
directories may vary. `chroma_db/` contains local persisted vector-store
data and should be treated as application data rather than source code.

### Main file responsibilities

  ---------------------------------------------------------------------------
  File                                    Responsibility
  --------------------------------------- -----------------------------------
  `app/ingestion.py`                      Loads PDF, Markdown, and text files
                                          and splits them into chunks.

  `app/vector_store.py`                   Configures embeddings and ChromaDB;
                                          indexes documents and removes
                                          document chunks.

  `app/retrieval.py`                      Implements dense retrieval, BM25
                                          caching, RRF fusion, CrossEncoder
                                          reranking, and retrieval-stage
                                          timing.

  `app/planning.py`                       Defines structured query planning
                                          and evidence assessment.

  `app/agent_tools.py`                    Implements web search, Wikipedia
                                          search, calculator, and the
                                          document RAG tool helper.

  `app/graph.py`                          Defines the LangGraph workflow and
                                          `answer_question()` entry point.

  `app/monitoring.py`                     Tracks request-level and
                                          stage-level metrics.

  `app/api.py`                            Defines FastAPI endpoints for
                                          health, metrics, uploads, document
                                          management, and questions.

  `app/streamlit_app.py`                  Implements the Streamlit frontend
                                          and calls the FastAPI backend.

  `evaluation/evaluate.py`                Runs RAGAS evaluation on the
                                          configured evaluation dataset.

  `evaluation/agent_evaluator.py`         Evaluates agent tool-selection
                                          behaviour against the agentic
                                          dataset.

  `evaluation/dataset.json`               Contains RAG evaluation examples.

  `evaluation/agentic_rag_dataset.json`   Contains agent/tool-selection
                                          evaluation examples.

  `requirements.txt`                      Lists Python dependencies.

  `.env.example`                          Documents required environment
                                          variables.
  ---------------------------------------------------------------------------

## Prerequisites

-   Python installed and available from the terminal.
-   Git, if cloning the repository.
-   A Google AI API key for Gemini.
-   A LangSmith API key if remote tracing is required.
-   Internet access for Gemini API requests, web search, Wikipedia
    access, and initial model downloads.

The development environment has used Python 3.13. Dependency
compatibility can vary by package version.

## Installation and Setup

The following commands assume Windows PowerShell and that the terminal
is opened at the repository root.

### 1. Clone the repository

``` powershell
git clone https://github.com/EshikaGupta/SentinelAI.git
cd SentinelAI
```

If the repository is already cloned, navigate to its local directory
instead.

### 2. Create and activate a virtual environment

``` powershell
py -m venv venv
.\venv\Scripts\Activate.ps1
```

### 3. Install dependencies

``` powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Configure environment variables

Create a `.env` file in the project root using `.env.example` as a
template.

``` dotenv
GOOGLE_API_KEY=your_google_api_key
GOOGLE_MODEL=your_configured_gemini_model

LANGSMITH_API_KEY=your_langsmith_api_key
LANGSMITH_TRACING=true
LANGSMITH_PROJECT=sentinel-ai
```

Use the model name configured for your Gemini account and current
application environment. Do not commit `.env` or real credentials.

## Running SentinelAI

Run commands from the project root so Python can import the `app`
package.

### 1. Start the FastAPI backend

``` powershell
uvicorn app.api:app --reload
```

FastAPI initializes the document index and BM25 cache during startup.

API documentation:

``` text
http://127.0.0.1:8000/docs
```

Health endpoint:

``` text
http://127.0.0.1:8000/health
```

Metrics endpoint:

``` text
http://127.0.0.1:8000/metrics
```

### 2. Start the Streamlit frontend

Open a second terminal, activate the same virtual environment, navigate
to the project root, and run:

``` powershell
streamlit run app/streamlit_app.py
```

The Streamlit application communicates with the FastAPI backend. Ensure
the API is running and the frontend's configured API URL points to it.

### 3. Use the API directly

Example request:

``` powershell
$body = @{
    question = "What does the internal documentation say about API rate limits?"
} | ConvertTo-Json

Invoke-RestMethod `
    -Uri "http://127.0.0.1:8000/ask" `
    -Method Post `
    -ContentType "application/json" `
    -Body $body
```

The response contains the question, answer, and sources.

## Monitoring and Observability

### LangSmith

Configure the LangSmith environment variables before starting the API:

``` dotenv
LANGSMITH_API_KEY=your_langsmith_api_key
LANGSMITH_TRACING=true
LANGSMITH_PROJECT=sentinel-ai
```

Use the LangSmith project to inspect supported model calls and
explicitly instrumented workflow/retrieval operations. Traces help
investigate model inputs and outputs, tool execution, retrieval
behaviour, latency, and failures.

### Custom metrics

The `/metrics` endpoint returns the custom in-memory metrics maintained
by `app/monitoring.py`.

The metrics include request totals, success and failure counts, average
request latency, returned-source counts, error counts, recent request
history, and recorded stage timings.

Metrics are process-local and reset when the API process restarts. They
are not a replacement for a persistent metrics backend in a multi-worker
deployment.

## Evaluation

### RAG evaluation

Run the RAGAS evaluation script from the project root:

``` powershell
python -m evaluation.evaluate
```

The script uses the configured evaluation dataset, invokes the
application, collects generated answers and retrieved contexts, and
evaluates the configured RAGAS metrics.

The current script configures context precision and context recall.
Results should be interpreted in the context of the dataset size and
evaluation configuration.

### Agent evaluation

Run the agent evaluator:

``` powershell
python -m evaluation.agent_evaluator
```

It runs the agentic evaluation examples and checks expected and
forbidden tool usage using the current tool-tracking implementation.

## Tests

Run the pytest suite from the project root:

``` powershell
python -m pytest -v
```

The repository also contains standalone diagnostic scripts for ChromaDB,
vector storage, or retrieval where present. Run those individually when
debugging the corresponding component.

## Environment Variables

  Variable              Purpose
  --------------------- ---------------------------------------------------
  `GOOGLE_API_KEY`      Authenticates Gemini API requests.
  `GOOGLE_MODEL`        Selects the Gemini model used by the application.
  `LANGSMITH_API_KEY`   Authenticates LangSmith.
  `LANGSMITH_TRACING`   Enables or disables LangSmith tracing.
  `LANGSMITH_PROJECT`   Sets the LangSmith project name.

Load credentials through environment variables. Never hard-code secrets
in Python files or commit them to the repository.

## Current Limitations

SentinelAI is a working development project, not a production-hardened
service.

-   **In-memory BM25 cache:** The BM25 retriever is process-local and
    must be refreshed from the indexed corpus when the application
    starts or the corpus changes.
-   **In-memory monitoring:** Custom metrics reset on restart and are
    not shared across multiple workers.
-   **Local inference overhead:** Embedding and CrossEncoder models run
    locally and can add CPU and memory overhead.
-   **External model dependency:** Gemini calls require network access,
    valid credentials, and available API quota.
-   **External search limitations:** Web search returns search-result
    information and snippets; the current implementation does not
    independently fetch and verify the full content of every result.
-   **Evaluation dataset size:** The current datasets are
    development-oriented and should be expanded before making broad
    quality or reliability claims.
-   **Agent evaluation tracking:** Document retrieval usage is inferred
    from the planner's decision, while external tool usage is extracted
    from tool messages. This is not yet a complete audit of every
    executed graph node or tool call.
-   **Process-local synchronization:** Python locks protect shared state
    within one process but do not coordinate multiple worker processes.
-   **Security:** Input validation and restricted calculator evaluation
    are useful safeguards, but they do not replace authentication,
    authorization, rate limiting, comprehensive file validation, or
    prompt-injection testing.
-   **Retrieval tuning:** Chunk sizes, candidate counts, and model
    choices are configuration values that should be validated against a
    larger representative evaluation set.

## Security Considerations

-   Never commit `.env`, API keys, or sensitive documents.
-   Treat uploaded documents and retrieved passages as untrusted
    content.
-   Do not allow instructions found inside retrieved documents or search
    results to override system instructions.
-   Validate incoming API inputs and restrict upload sizes and supported
    file types.
-   Avoid logging credentials or sensitive user information.
-   Treat LLM-based evaluations as advisory rather than as security
    guarantees.
-   Add authentication, authorization, and rate limiting before exposing
    private document collections to multiple users or the public
    internet.

## Project Goal

SentinelAI brings together document ingestion, hybrid retrieval,
reranking, evidence assessment, agentic tool use, grounded answer
generation, tracing, monitoring, and evaluation in one application.

Its focus is not only on generating answers, but also on making the
retrieval and reasoning workflow inspectable and measurable.
