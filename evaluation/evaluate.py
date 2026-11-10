import json
import os
from pathlib import Path
from langchain_google_genai import ChatGoogleGenerativeAI
from ragas.llms import LangchainLLMWrapper
from dotenv import load_dotenv
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from app.graph import answer_question
from ragas.run_config import RunConfig

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = PROJECT_ROOT / "evaluation" / "dataset.json"

load_dotenv(PROJECT_ROOT / ".env")


def answer_to_text(answer) -> str:
    """Normalize LangChain's possible string/content-block answer formats."""
    if isinstance(answer, str):
        return answer

    if isinstance(answer, list):
        parts = []
        for item in answer:
            if isinstance(item, dict) and isinstance(item.get("text"), str):
                parts.append(item["text"])
            elif isinstance(item, str):
                parts.append(item)
        return "\n".join(parts)

    return str(answer)


def main():
    cases = json.loads(DATASET_PATH.read_text(encoding="utf-8"))

    questions = []
    answers = []
    contexts = []
    ground_truths = []

    for case in cases:
        result = answer_question(case["question"])

        questions.append(case["question"])
        answers.append(answer_to_text(result["answer"]))
        contexts.append([
            source["text"]
            for source in result.get("sources", [])
            if source.get("text")
        ])
        ground_truths.append(case["ground_truth"])

    dataset = Dataset.from_dict({
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths,
    })



    evaluator_llm = LangchainLLMWrapper(
        ChatGoogleGenerativeAI(
            model=os.getenv("RAGAS_MODEL", "gemini-3.6-flash")
        )
    )
    evaluator_embeddings = GoogleGenerativeAIEmbeddings(
        model=os.getenv("RAGAS_EMBEDDING_MODEL", "models/gemini-embedding-001")
    )
    from ragas.metrics import Faithfulness,AnswerRelevancy
    
    faithfulness_metric = Faithfulness(
        llm=evaluator_llm
    )

    answer_relevancy_metric = AnswerRelevancy(
        llm=evaluator_llm,
        embeddings=evaluator_embeddings,
        strictness=1
    )

    results = evaluate(
        dataset,
        metrics=[
            # faithfulness_metric,
            # answer_relevancy_metric,
            context_precision,
            context_recall,
        ],
        run_config=RunConfig(
        timeout=300,
        max_retries=2,
        max_workers=1,
        ),
        llm=evaluator_llm,embeddings=evaluator_embeddings,
    )

    print(results)
    results.to_pandas().to_csv(
        PROJECT_ROOT / "evaluation" / "ragas_results.csv",
        index=False,
    )


if __name__ == "__main__":
    main()