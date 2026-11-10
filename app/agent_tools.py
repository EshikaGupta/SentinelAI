
import ast
import operator
import json
from pathlib import Path

from ddgs import DDGS
import wikipedia

from langchain_core.tools import tool

from .ingestion import load_document, split_document
from .vector_store import get_vector_store, index_directory
from .retrieval import retrieve_and_rerank


# --------------------------------------------------
# 1. DOCUMENT RAG TOOL
# --------------------------------------------------

def create_document_rag_tool(source_path: str):

    @tool
    def document_rag(query: str) -> str:
        """
        Search SentinelAI's indexed documents using hybrid retrieval
        and cross-encoder reranking.

        Use this tool for questions about uploaded documents,
        internal documentation, or indexed knowledge.
        """
        directory = Path(source_path).resolve()

        if not directory.exists():
            return "Document directory does not exist."

        # Keep the index synchronized with the document directory.
        index_directory(directory)
        vector_store = get_vector_store()

        chunks = []
        supported = {".txt", ".md", ".pdf"}

        for path in sorted(directory.rglob("*")):
            if path.is_file() and path.suffix.lower() in supported:
                text = load_document(path)
                chunks.extend(
                    split_document(
                        text,
                        source=str(path.resolve())
                    )
                )

        documents = retrieve_and_rerank(
            query=query,
            chunks=chunks,
            vector_store=vector_store,
            source=None,
            top_k=5,
        )

        if not documents:
            return json.dumps({
                "found": False,
                "message": "No relevant information found in the documents.",
                "sources": [],
            })

        sources = []

        for index, doc in enumerate(documents, start=1):
            sources.append({
                "label": f"Source {index}",
                "source": doc.metadata.get("source", "unknown"),
                "chunk_index": doc.metadata.get("chunk_index"),
                "text": doc.page_content,
                "rerank_score": doc.metadata.get("rerank_score"),
            })

        return json.dumps({
            "found": True,
            "message": "Relevant document passages retrieved.",
            "sources": sources,
        }, ensure_ascii=False)

    return document_rag


# --------------------------------------------------
# 2. WEB SEARCH TOOL
# --------------------------------------------------

@tool
def web_search(query: str) -> str:
    """
    Search the web for current information, recent developments,
    news, technical documentation, and online sources.

    Use this when information may be recent or is not available
    in the internal document collection.
    """
    try:
        results = DDGS().text(
            query,
            max_results=5,
        )

        results = list(results)

        if not results:
            return "No web search results found."

        formatted = []

        for index, result in enumerate(results, start=1):
            formatted.append({
                "result": index,
                "title": result.get("title"),
                "url": result.get("href"),
                "snippet": result.get("body"),
            })

        return json.dumps(formatted, ensure_ascii=False)

    except Exception as exc:
        return f"Web search failed: {str(exc)}"


# --------------------------------------------------
# 3. WIKIPEDIA TOOL
# --------------------------------------------------

@tool
def wikipedia_search(query: str) -> str:
    """
    Search Wikipedia for general knowledge, definitions,
    concepts, people, technologies, and background information.
    """
    try:
        search_results = wikipedia.search(query, results=3)

        if not search_results:
            return "No relevant Wikipedia pages found."

        pages = []

        for title in search_results[:3]:
            try:
                page = wikipedia.page(
                    title,
                    auto_suggest=False,
                    redirect=True,
                )

                pages.append({
                    "title": page.title,
                    "url": page.url,
                    "summary": wikipedia.summary(
                        page.title,
                        sentences=5,
                        auto_suggest=False,
                    ),
                })

            except (
                wikipedia.exceptions.DisambiguationError,
                wikipedia.exceptions.PageError,
            ):
                continue

        if not pages:
            return "No usable Wikipedia pages found."

        return json.dumps(pages, ensure_ascii=False)

    except Exception as exc:
        return f"Wikipedia search failed: {str(exc)}"


# --------------------------------------------------
# 4. CALCULATOR TOOL
# --------------------------------------------------

_ALLOWED_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
    ast.FloorDiv: operator.floordiv,
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


def _safe_calculate(node):
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return node.value
        raise ValueError("Only numeric values are allowed.")

    if isinstance(node, ast.BinOp):
        op = _ALLOWED_OPERATORS.get(type(node.op))

        if op is None:
            raise ValueError("Unsupported mathematical operation.")

        left = _safe_calculate(node.left)
        right = _safe_calculate(node.right)

        if isinstance(node.op, ast.Pow) and abs(right) > 100:
            raise ValueError("Exponent is too large.")

        result = op(left, right)

        if abs(result) > 1e100:
            raise ValueError("Result is too large.")

        return result

    if isinstance(node, ast.UnaryOp):
        op = _ALLOWED_OPERATORS.get(type(node.op))

        if op is None:
            raise ValueError("Unsupported operation.")

        return op(_safe_calculate(node.operand))

    raise ValueError("Invalid mathematical expression.")


@tool
def calculator(expression: str) -> str:
    """
    Evaluate mathematical expressions involving addition,
    subtraction, multiplication, division, powers, modulo,
    and parentheses.

    Use this tool whenever an accurate numerical calculation
    is needed instead of calculating mentally.
    """
    try:
        if len(expression) > 200:
            return "Expression is too long."

        tree = ast.parse(expression, mode="eval")
        result = _safe_calculate(tree.body)

        return json.dumps({
            "expression": expression,
            "result": result,
        })

    except Exception as exc:
        return f"Calculation failed: {str(exc)}"
