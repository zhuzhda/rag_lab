import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rag_lab.chunking import chunk_documents
from rag_lab.corpus import load_documents
from rag_lab.embeddings import TfidfEmbedder
from rag_lab.evaluation import evaluate, load_questions
from rag_lab.retrieval import Retriever

KS = [1, 3, 5, 10]


def run(chunk_size: int, overlap: int, documents, questions):
    chunks = chunk_documents(documents, chunk_size=chunk_size, overlap=overlap)
    result = evaluate(Retriever(chunks, TfidfEmbedder()), questions, ks=KS)
    lengths = [chunk.token_count for chunk in chunks]
    return {
        "chunk_size": chunk_size, 
        "overlap": overlap,
        "hit@1": result["aggregate"]["1"]["hit"],
        "hit@3": result["aggregate"]["3"]["hit"],
        "hit@5": result["aggregate"]["5"]["hit"],
        "recall@5": result["aggregate"]["5"]["recall"],
        "precision@5": result["aggregate"]["5"]["precision"],
        "mrr@5": result["aggregate"]["5"]["reciprocal_rank"],
        "chunks": len(chunks), 
        "avg_length": round(sum(lengths) / len(lengths), 2),
    }


def main():
    documents = load_documents(ROOT / "documents")
    questions = load_questions(ROOT / "data" / "dev_questions.json")
    baseline = [run(120, 20, documents, questions)]
    by_size = [run(size, 0, documents, questions) for size in (60, 100, 160)]
    best = max(by_size, key=lambda row: row["mrr@5"])["chunk_size"]
    by_overlap = [run(best, overlap, documents, questions) for overlap in (20, 40)]
    rows = baseline + by_size + by_overlap

    with open(ROOT / "results" / "chunking.json", "w") as file:
        json.dump(rows, file, indent=2)
    for row in rows:
        print(row)


if __name__ == "__main__":
    main()
