
import json
import logging
from pathlib import Path
from datetime import datetime
from app.vector_store import index_directory, DOCUMENTS_DIRECTORY
from app.retrieval import refresh_bm25_cache
from app.graph import answer_question

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATASET_PATH = (
    PROJECT_ROOT
    / "evaluation"
    / "agentic_rag_dataset.json"
)

RESULTS_DIR = PROJECT_ROOT / "evaluation" / "results"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def initialize_evaluation():

    logger.info("Initializing SentinelAI for evaluation...")

    # Synchronize documents with persistent Chroma.
    index_directory(DOCUMENTS_DIRECTORY)

    # Build the in-memory BM25 cache.
    count = refresh_bm25_cache()

    logger.info(
        "Evaluation ready. BM25 chunks loaded: %d",
        count,
    )

def load_dataset():
    with open(DATASET_PATH, "r", encoding="utf-8") as file:
        return json.load(file)


def evaluate_case(test_case):

    question = test_case["question"]

    expected_tools = set(test_case.get("required_tools", []))
    forbidden_tools = set(test_case.get("forbidden_tools", []))

    logger.info("Evaluating %s", test_case["id"])

    try:
        response = answer_question(question)

        actual_tools = set(response.get("tools_used", []))

        missing_tools = expected_tools - actual_tools
        forbidden_used = forbidden_tools & actual_tools
        unexpected_tools = actual_tools - expected_tools

        required_coverage = (
            len(expected_tools & actual_tools) / len(expected_tools)
            if expected_tools
            else 1.0
        )

        passed = (
            not missing_tools
            and not forbidden_used
        )

        return {
            "id": test_case["id"],
            "question": question,
            "category": test_case["category"],
            "expected_tools": sorted(expected_tools),
            "actual_tools": sorted(actual_tools),
            "missing_tools": sorted(missing_tools),
            "forbidden_tools_used": sorted(forbidden_used),
            "unexpected_tools": sorted(unexpected_tools),
            "required_tool_coverage": round(required_coverage, 2),
            "passed": passed,
            "answer": response.get("answer"),
            "sources": response.get("sources", []),
            "error": None,
        }

    except Exception as exc:
        logger.exception("Evaluation failed: %s", test_case["id"])

        return {
            "id": test_case["id"],
            "question": question,
            "category": test_case["category"],
            "expected_tools": sorted(expected_tools),
            "actual_tools": [],
            "missing_tools": sorted(expected_tools),
            "forbidden_tools_used": [],
            "unexpected_tools": [],
            "required_tool_coverage": 0.0,
            "passed": False,
            "answer": None,
            "sources": [],
            "error": str(exc),
        }


def calculate_metrics(results):

    total = len(results)

    if total == 0:
        return {}

    successful = [r for r in results if r["error"] is None]

    passed = sum(r["passed"] for r in results)

    total_required = sum(
        len(r["expected_tools"])
        for r in results
    )

    total_missing = sum(
        len(r["missing_tools"])
        for r in results
    )

    total_forbidden = sum(
        len(r["forbidden_tools_used"])
        for r in results
    )

    return {
        "total_questions": total,
        "successful_executions": len(successful),
        "execution_failure_rate": round(
            (total - len(successful)) / total, 3
        ),
        "tool_selection_accuracy": round(passed / total, 3),
        "required_tool_recall": round(
            (total_required - total_missing) / total_required,
            3,
        ) if total_required else 1.0,
        "forbidden_tool_usage_count": total_forbidden,
        "pass_count": passed,
        "fail_count": total - passed,
    }


def run_evaluation():
    initialize_evaluation()
    dataset = load_dataset()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    results = []

    for test_case in dataset:
        result = evaluate_case(test_case)
        results.append(result)

        logger.info(
            "%s | Passed: %s | Expected: %s | Actual: %s",
            result["id"],
            result["passed"],
            result["expected_tools"],
            result["actual_tools"],
        )

    metrics = calculate_metrics(results)

    output = {
        "evaluated_at": datetime.now().isoformat(),
        "metrics": metrics,
        "results": results,
    }

    output_path = RESULTS_DIR / "agent_evaluation_results.json"

    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(output, file, indent=4, ensure_ascii=False, default=str)

    print("\n========== AGENT EVALUATION ==========")
    print(json.dumps(metrics, indent=4))
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    run_evaluation()
