import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rag_lab.chunking import chunk_documents
from rag_lab.corpus import load_documents
from rag_lab.embeddings import TfidfEmbedder
from rag_lab.evaluation import evaluate, load_questions
from rag_lab.retrieval import Retriever

K = 5


def ranked(key, chunk_size, overlap, documents, questions):
    chunks = chunk_documents(documents, chunk_size=chunk_size, overlap=overlap)
    report = evaluate(Retriever(chunks, TfidfEmbedder()), questions, ks=[K])
    return {row["id"]: [entry[key] for entry in row["retrieved"]] for row in report["questions"]}


def diff(label, before, after, questions, limit):
    relevant = {q["id"]: q["relevant_doc_ids"] for q in questions}
    shown = 0
    for question in questions:
        question_id = question["id"]
        if before[question_id][:K] == after[question_id][:K]:
            continue
        print(f"[{label}] {question_id}: {question['question']}")
        print(f"  было:  {before[question_id][:K]}")
        print(f"  стало: {after[question_id][:K]}  gold={relevant[question_id]}")
        shown += 1
        if shown == limit:
            return


def main():
    documents = load_documents(ROOT / "documents")
    questions = load_questions(ROOT / "data" / "dev_questions.json")
    diff(
        "chunk_size 60->160",
        ranked("document_id", 60, 0, documents, questions),
        ranked("document_id", 160, 0, documents, questions),
        questions,
        3,
    )
    diff(
        "overlap 0->40",
        ranked("chunk_id", 100, 0, documents, questions),
        ranked("chunk_id", 100, 40, documents, questions),
        questions,
        2,
    )


if __name__ == "__main__":
    main()
