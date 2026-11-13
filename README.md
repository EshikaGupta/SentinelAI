# SentinelAI --- Agentic RAG with Hybrid Retrieval and Observability

SentinelAI is an agentic Retrieval-Augmented Generation (RAG)
application that answers questions using internal documents and, when
needed, external information sources. It combines retrieval-first
execution, hybrid retrieval, evidence assessment, tool-calling, and
answer generation in a stateful LangGraph workflow.

The application supports PDF, Markdown, and text documents. It uses
ChromaDB for persistent vector storage, BM25 for keyword retrieval,
Reciprocal Rank Fusion (RRF) to combine rankings, and a CrossEncoder to
rerank candidate passages. Google Gemini powers external-intent
planning, evidence assessment, research/tool selection, and answer
generation. The planner can use a separately configured Gemini model.

SentinelAI also includes a FastAPI backend, a Streamlit interface,
LangSmith tracing, custom request and retrieval monitoring, and
evaluation scripts for RAG quality and agent tool selection.

## Key Features

### 1. Document ingestion and chunking

-   Supports `.pdf`, `.md`, and `.txt` documents.
-   Uses LangChain's `RecursiveCharacterTextSplitter`.
-   Preserves source metadata and chunk positions.
-   Uses overlapping chunks to retain context across chunk boundaries.

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
-   Supports re-indexing, stale-chunk removal, and document deletion.

### 3. Hybrid retrieval

SentinelAI combines:

-   **Dense retrieval:** Embedding similarity search using ChromaDB.
-   **Sparse retrieval:** BM25 keyword retrieval.

The BM25 corpus is loaded from ChromaDB and cached in memory. A corpus
fingerprint determines whether the cache needs refreshing. Separate
cached BM25 retrievers are maintained for different source filters and
`k` values.

### 4. Reciprocal Rank Fusion

Dense and BM25 retrieval produce rankings with different score scales.
SentinelAI combines their rankings using Reciprocal Rank Fusion (RRF),
which uses ranking positions rather than adding raw retrieval scores.

The current RRF constant is `60`.

\[ RRF(d)=`\sum`{=tex}\_{r
`\in `{=tex}R}`\frac{1}{k_0+\operatorname{rank}_r(d)}`{=tex} \]

### 5. CrossEncoder reranking and retrieval signals

A CrossEncoder scores query-document pairs and reorders candidate
passages.

Model: `cross-encoder/ms-marco-MiniLM-L-6-v2`

Current configuration:

``` python
DENSE_K = 10
SPARSE_K = 10
RERANK_TOP_K = 5
RRF_CONSTANT = 60
```

The retrieval pipeline preserves these metadata fields:

-   `dense_score` and `dense_rank`
-   `sparse_rank`
-   `rrf_score` and `rrf_rank`
-   `rerank_score` and `rerank_rank`
-   `rerank_top_score`
-   `rerank_score_gap`
-   `rerank_score_mean`
-   `rerank_score_std`

These signals support evidence evaluation and debugging. They are not
calibrated confidence probabilities, and no individual score proves that
a document contains the answer.

### 6. Retrieval-first agentic workflow with LangGraph

The workflow follows a retrieval-first architecture:

1.  **Parallel execution:** Internal document retrieval and
    external-intent planning start independently.
2.  **Document retrieval:** Hybrid retrieval and CrossEncoder reranking
    always execute for every query.
3.  **Evidence evaluation:** Assesses whether retrieved passages
    sufficiently cover the question, partially cover it, or are
    insufficient.
4.  **Conditional routing:** If evidence is sufficient and no additional
    external information is required, the graph generates a
    document-grounded answer. Otherwise, it invokes the research agent.
5.  **Research agent:** Uses Gemini tool calling to decide whether
    external research or calculation is needed.
6.  **Tool execution:** LangGraph's `ToolNode` executes requested
    external tools and returns results to the research agent.
7.  **Finalization:** Produces a final response when the research
    iteration limit is reached.

The retrieval and planning branches are synchronized before evidence
evaluation. The planner does not decide whether internal documents
should be searched.

The external-intent planner uses structured output through a separately
configurable Gemini model. It identifies requirements for web search,
Wikipedia, and calculations. The planner prompt selects Wikipedia only
when explicitly requested or when encyclopedic external background is
explicitly requested. General educational or historical questions do not
automatically require Wikipedia.

The graph's routing logic treats explicit web-search and calculator
requirements separately from local evidence sufficiency. A Wikipedia
selection alone does not override a sufficient internal-document answer.

External tools:

  -----------------------------------------------------------------------
  Tool                                Purpose
  ----------------------------------- -----------------------------------
  `web_search`                        Searches the web using DDGS and
                                      returns titles, URLs, and snippets.

  `wikipedia_search`                  Searches Wikipedia and returns page
                                      summaries and URLs.

  `calculator`                        Evaluates supported mathematical
                                      expressions using a restricted
                                      AST-based evaluator.
  -----------------------------------------------------------------------

Internal document retrieval is a dedicated LangGraph node. The
`document_rag` helper exists in `app/agent_tools.py`, but is not bound
to the current research agent.

The current graph configures `MAX_RESEARCH_ITERATIONS = 3`.

### 7. Evidence assessment

The evaluator receives the question, retrieved passages and source
metadata, dense retrieval scores and ranks, BM25 ranks, RRF scores and
ranks, CrossEncoder scores and ranks, and reranker score-distribution
information.

Gemini returns a structured assessment:

-   `status`: `sufficient`, `partial`, or `insufficient`.
-   `covered_parts`: Parts supported by the retrieved evidence.
-   `missing_parts`: Parts not adequately supported.
-   `reason`: Explanation of the assessment.

Retrieval signals support the assessment, but document content is the
primary basis for factual coverage. Scores are not treated as calibrated
probabilities, and arbitrary numerical confidence thresholds are not
used.

### 8. Answer and source handling

-   Generates answers using Google Gemini through LangChain.
-   Uses retrieved passages for document-grounded answers.
-   Passes internal evidence and the evidence assessment to the research
    agent when external research is required.
-   Instructs document-grounded answers to acknowledge evidence gaps.
-   Extracts and deduplicates internal and external source information.
-   Preserves retrieval metadata in internal document source records.

The answer is normalized into a `list[dict]` structure, with each answer
block containing a `text` field, maintaining compatibility with the
Streamlit frontend.

Retrieved document text and external search results are treated as
information to evaluate, not instructions that override system
behaviour.

### 9. LangSmith tracing

LangSmith configuration is loaded from environment variables.
`@traceable` instrumentation covers important workflow and retrieval
operations, including planning, document retrieval, evidence evaluation,
answer generation, research, finalization, dense retrieval, BM25
retrieval, RRF, and reranking.

Tracing helps inspect supported run inputs and outputs, execution
structure, latency, and failures. Exact trace coverage depends on
integrations and explicit instrumentation.

### 10. Custom monitoring and request logging

`app/monitoring.py` provides lightweight in-process monitoring.

Request-level metrics include:

-   Total, successful, and failed requests.
-   Total and average request latency.
-   Average number of returned sources.
-   Error counts by exception type.
-   A bounded history of recent requests.

Stage-level metrics record execution counts, cumulative duration,
average duration, and supplied details for instrumented stages such as
hybrid retrieval and reranking.

The graph logs start and completion times for planning, retrieval,
evidence evaluation, document answer generation, research, and
finalization. The API also logs received questions and successful
request duration.

Metrics are held in memory. They reset when the process restarts and are
not shared across multiple worker processes.

### 11. FastAPI backend

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

The `/ask` endpoint validates the question and returns the question,
answer, and sources. It logs the received question and total successful
request execution time. Uploads are streamed and limited to 10 MB. The
API initializes the document index and BM25 cache during startup.

### 12. Streamlit interface

`app/streamlit_app.py` communicates with FastAPI over HTTP and provides:

-   Conversational question answering and chat history.
-   New-chat handling.
-   Document upload.
-   Document listing and deletion.
-   Answer and source display.

The `/ask` request uses a 10-second connection timeout and a 300-second
read timeout.

### 13. Evaluation

#### RAGAS evaluation

`evaluation/evaluate.py` runs questions from the evaluation dataset
through the application, collects generated answers and retrieved
contexts, and evaluates configured RAGAS metrics.

The current script configures context precision and context recall.
Results depend on the dataset, model, configuration, and evaluation run;
the current dataset is suitable for development checks, not broad
statistical claims.

#### Agent evaluation

`evaluation/agent_evaluator.py` checks expected and forbidden tool usage
and reports measures such as required-tool recall, tool-selection
accuracy, forbidden-tool usage, and execution failures.

The current evaluation uses the recorded plan and tool messages.
Document retrieval is inferred from the planner's document requirement,
while external tool usage is identified from tool messages. It is not a
complete independent audit of every graph-node execution.

### 14. Tests

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
    D --> E["LangGraph"]

    E --> F["Document Retrieval"]
    E --> G["External-Intent Planner"]

    F --> H["ChromaDB Dense Search"]
    F --> I["BM25 Sparse Search"]

    H --> J["RRF Fusion"]
    I --> J
    J --> K["CrossEncoder Reranking"]

    G --> L["Planning Result"]

    K --> M["Evidence Evaluator"]
    L --> M

    M --> N{"Evidence sufficient and no external requirement?"}
    N -->|Yes| O["Document Answer Generation"]
    N -->|No| P["Research Agent"]

    P --> Q{"External tool call?"}
    Q -->|Yes| R["LangGraph ToolNode"]
    R --> P
    Q -->|No| S["Research Answer"]
    P --> T["Finalize on iteration limit"]

    O --> U["Extract Answer and Sources"]
    S --> U
    T --> U
    U --> V["FastAPI Response"]

    D -. "Request and stage metrics" .-> W["Custom Monitoring"]
    D -. "Traces" .-> X["LangSmith"]
```

LangSmith tracing inspects individual executions, while custom
monitoring aggregates request outcomes and stage timings.

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
                                                  parallel execution,
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
SentinelAI/
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
├── data/
│   └── sample_docs/
├── evaluation/
│   ├── agent_evaluator.py
│   ├── agentic_rag_dataset.json
│   ├── dataset.json
│   └── evaluate.py
├── tests/
├── chroma_db/
├── .env.example
├── .gitignore
├── requirements.txt
└── README.md
```

Local runtime directories and dataset contents may vary. `chroma_db/`
contains persisted vector-store data.

### Main file responsibilities

  -----------------------------------------------------------------------
  File                                Responsibility
  ----------------------------------- -----------------------------------
  `app/ingestion.py`                  Loads PDF, Markdown, and text files
                                      and splits them into chunks.

  `app/vector_store.py`               Configures embeddings and ChromaDB;
                                      indexes documents and removes
                                      document chunks.

  `app/retrieval.py`                  Implements dense retrieval, BM25
                                      caching, RRF fusion, CrossEncoder
                                      reranking, retrieval metadata, and
                                      retrieval-stage timing.

  `app/planning.py`                   Defines structured external-intent
                                      planning, retrieval-signal
                                      extraction, and evidence
                                      assessment.

  `app/agent_tools.py`                Implements web search, Wikipedia
                                      search, calculator, and the
                                      document RAG tool helper.

  `app/graph.py`                      Defines the parallel
                                      retrieval-first LangGraph workflow,
                                      conditional routing, node-level
                                      timing logs, research loop, and
                                      `answer_question()`.

  `app/monitoring.py`                 Tracks request-level and
                                      stage-level metrics.

  `app/api.py`                        Defines FastAPI endpoints for
                                      health, metrics, uploads, document
                                      management, and questions.

  `app/streamlit_app.py`              Implements the Streamlit frontend
                                      and calls the FastAPI backend.

  `evaluation/evaluate.py`            Runs RAGAS evaluation on the
                                      configured evaluation dataset.

  `evaluation/agent_evaluator.py`     Evaluates agent tool-selection
                                      behaviour against the agentic
                                      dataset.

  `requirements.txt`                  Lists Python dependencies.

  `.env.example`                      Documents environment variables.
  -----------------------------------------------------------------------

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

# Main model used for evidence evaluation, research, and answer generation
GOOGLE_MODEL=gemini-3.6-flash

# Optional: smaller model used for external-intent planning
# Defaults to gemini-3.5-flash-lite when not specified
PLANNER_MODEL=gemini-3.5-flash-lite

LANGSMITH_API_KEY=your_langsmith_api_key
LANGSMITH_TRACING=true
LANGSMITH_PROJECT=sentinel-ai
```

`GOOGLE_MODEL` selects the main Gemini model. `PLANNER_MODEL` selects
the model used for structured external-intent planning. If
`PLANNER_MODEL` is omitted, the application defaults to
`gemini-3.5-flash-lite`.

Use model names available to your Google AI API account and current
application environment. Free-tier access and quotas depend on model
availability and account limits. Do not commit `.env` or real
credentials.

## Running SentinelAI

Run commands from the project root so Python can import the `app`
package.

### 1. Start the FastAPI backend

``` powershell
uvicorn app.api:app --reload
```

FastAPI initializes the document index and BM25 cache during startup.

API documentation: `http://127.0.0.1:8000/docs`

Health endpoint: `http://127.0.0.1:8000/health`

Metrics endpoint: `http://127.0.0.1:8000/metrics`

### 2. Start the Streamlit frontend

Open a second terminal, activate the same virtual environment, navigate
to the project root, and run:

``` powershell
streamlit run app/streamlit_app.py
```

Ensure the API is running and the frontend's configured API URL points
to it.

### 3. Use the API directly

Example PowerShell request:

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
explicitly instrumented workflow/retrieval operations.

### Custom metrics and logs

The `/metrics` endpoint returns the custom in-memory metrics maintained
by `app/monitoring.py`.

The metrics include request totals, success and failure counts, average
request latency, returned-source counts, error counts, recent request
history, and recorded stage timings.

The FastAPI `/ask` endpoint logs the received question and total
successful request duration. LangGraph nodes also log their start and
completion times.

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
evaluates the configured RAGAS metrics. The current script configures
context precision and context recall.

### Agent evaluation

Run the agent evaluator:

``` powershell
python -m evaluation.agent_evaluator
```

It checks expected and forbidden tool usage using the current
tool-tracking implementation.

## Tests

Run the pytest suite from the project root:

``` powershell
python -m pytest -v
```

## Environment Variables

  -----------------------------------------------------------------------
  Variable                            Purpose
  ----------------------------------- -----------------------------------
  `GOOGLE_API_KEY`                    Authenticates Gemini API requests.

  `GOOGLE_MODEL`                      Selects the main Gemini model used
                                      for evidence evaluation, research,
                                      and answer generation.

  `PLANNER_MODEL`                     Selects the Gemini model used for
                                      structured external-intent
                                      planning. Optional; defaults to
                                      `gemini-3.5-flash-lite`.

  `LANGSMITH_API_KEY`                 Authenticates LangSmith.

  `LANGSMITH_TRACING`                 Enables or disables LangSmith
                                      tracing.

  `LANGSMITH_PROJECT`                 Sets the LangSmith project name.
  -----------------------------------------------------------------------

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
-   **Planner classification:** External-intent planning uses structured
    LLM output. Its decisions depend on the configured planner model and
    prompt and can occasionally select or miss an external tool.
-   **Security:** Input validation and restricted calculator evaluation
    are useful safeguards, but they do not replace authentication,
    authorization, rate limiting, comprehensive file validation, or
    prompt-injection testing.
-   **Retrieval tuning:** Chunk sizes, candidate counts, and model
    choices should be validated against a larger representative
    evaluation set.

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
