from __future__ import annotations

import json
from pathlib import Path

from dotenv import load_dotenv
from datasets import Dataset
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings
from ragas import evaluate
from ragas.metrics import Faithfulness, AnswerRelevancy, ContextPrecision, ContextRecall
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

from app.core.config import settings
from app.core.logging import setup_logging
from app.pipeline.rag_pipeline import RAGPipeline
"""
RAGAS evaluation — runs the RAG pipeline against test questions and scores it.
"""

load_dotenv()
setup_logging()

_QUESTIONS_FILE = Path(__file__).parent / "test_question.json"


def load_questions() -> list[dict]:
    with _QUESTIONS_FILE.open(encoding="utf-8") as f:
        return json.load(f)


def run_rag_pipeline(pipeline: RAGPipeline, questions: list[dict]) -> Dataset:
    data: dict[str, list] = {
        "question": [],
        "answer": [],
        "contexts": [],
        "ground_truth": [],
    }

    for item in questions:
        q = item["question"]
        gt = item["ground_truth"]
        print(f"  Running: {q}")

        docs = pipeline.retrieve(q)
        context = pipeline._prepare_context(docs)
        answer = pipeline.generate(q, context)

        data["question"].append(q)
        data["answer"].append(answer)
        data["contexts"].append([doc["content"] for doc in docs])
        data["ground_truth"].append(gt)

    return Dataset.from_dict(data)


def main() -> None:
    print("Initialising RAG pipeline …")
    pipeline = RAGPipeline()
    pipeline.initialize_database()

    questions = load_questions()
    print(f"Loaded {len(questions)} test questions")

    dataset = run_rag_pipeline(pipeline, questions)

    print("\nRunning RAGAS evaluation …\n")

    evaluator_llm = LangchainLLMWrapper(
        ChatGroq(model=settings.LLM_MODEL, temperature=0)
    )
    ragas_embeddings = LangchainEmbeddingsWrapper(
        HuggingFaceEmbeddings(model_name=settings.EMBEDDING_MODEL)
    )

    result = evaluate(
        dataset,
        metrics=[
            Faithfulness(),
            AnswerRelevancy(),
            ContextPrecision(),
            ContextRecall(),
        ],
        llm=evaluator_llm,
        embeddings=ragas_embeddings,
    )

    print("\nEvaluation Results:")
    print(result)


if __name__ == "__main__":
    main()