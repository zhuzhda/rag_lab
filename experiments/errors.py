import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rag_lab.chunking import chunk_documents
from rag_lab.corpus import load_documents
from rag_lab.embeddings import TfidfEmbedder
from rag_lab.evaluation import evaluate, load_questions
from rag_lab.retrieval import Retriever

CHUNK_SIZE = 100
OVERLAP = 0
K = 5
FIELDS = [
    "question_id",
    "question",
    "relevant_doc_ids",
    "top_retrieved_doc_ids",
    "error_class",
    "explanation",
    "proposed_fix",
]


def main():
    documents = load_documents(ROOT / "documents")
    questions = load_questions(ROOT / "data" / "dev_questions.json")
    chunks = chunk_documents(documents, chunk_size=CHUNK_SIZE, overlap=OVERLAP)
    report = evaluate(Retriever(chunks, TfidfEmbedder()), questions, ks=[1, K])

    rows = []
    for row in report["questions"]:
        metrics = row["metrics"]
        if metrics["1"]["hit"] != 0.0:
            continue
        error_class = ""
        if metrics[str(K)]["hit"] == 0.0:
            error_class = "полный промах"
        else:
            error_class = "низкий ранг"
        rows.append(
            {
                "question_id": row["id"],
                "question": row["question"],
                "relevant_doc_ids": ";".join(row["relevant_doc_ids"]),
                "top_retrieved_doc_ids": ";".join(
                    entry["document_id"] for entry in row["retrieved"][:K]
                ),
                "error_class": error_class,
                "explanation": "",
                "proposed_fix": "",
            }
        )

    with open(ROOT / "templates" / "error_analysis.csv", "w", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
        
    for row in rows:
        print(row["question_id"], row["error_class"])


if __name__ == "__main__":
    main()
