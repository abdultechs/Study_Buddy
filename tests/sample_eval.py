import os
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.document_loader import DocumentProcessor
from src.vector_store import VectorStoreManager

SAMPLE_PDF_PATH = "data/sample_docs/sample_policy.pdf"

# Benchmark queries grounded in data/sample_docs/sample_policy.pdf
IN_DOCUMENT_QUERIES = [
    {
        "query": "What are the core collaboration hours for remote team members?",
        "expected_page": 1,
    },
    {
        "query": "What is the daily meal per diem limit for business travel?",
        "expected_page": 2,
    },
    {
        "query": "How many days of annual paid time off (PTO) do employees accrue?",
        "expected_page": 3,
    },
    {
        "query": "How quickly must lost or stolen hardware be reported to IT security?",
        "expected_page": 4,
    },
    {
        "query": "What is the nightly lodging ceiling for hotels in metropolitan areas?",
        "expected_page": 2,
    },
    {
        "query": "How many weeks of paid parental leave are primary caregivers entitled to?",
        "expected_page": 3,
    },
]

# Off-topic / adversarial queries that MUST NOT be grounded in the document
OFF_TOPIC_QUERIES = [
    "What is the average surface atmospheric pressure on Mars?",
    "How do I bake traditional chocolate chip cookies from scratch?",
    "What were the company's gross financial earnings in 1982?",
    "Who scored the winning goal in the 1998 FIFA World Cup final?",
]


def run_evaluation():
    print("=" * 75)
    print("  RAG RETRIEVAL & EMPIRICAL THRESHOLD EVALUATION BENCHMARK")
    print("=" * 75)

    if not Path(SAMPLE_PDF_PATH).exists():
        print(f"Error: Sample PDF '{SAMPLE_PDF_PATH}' not found. Please generate it first.")
        return

    # Ingest sample PDF
    with open(SAMPLE_PDF_PATH, "rb") as f:
        pdf_bytes = f.read()

    processor = DocumentProcessor()
    chunks, doc_info = processor.process_pdf(pdf_bytes, "sample_policy.pdf")
    print(f"\n[1/3] Ingested '{doc_info.filename}': {doc_info.page_count} pages, {len(chunks)} chunks.")

    # Initialize Vector Store
    vstore = VectorStoreManager()
    vstore.upsert_chunks(chunks)
    print(f"[2/3] Vector store index ready (Collection: {vstore.collection.name}).\n")

    # Evaluate In-Document Queries
    print("-" * 75)
    print("  EVALUATING IN-DOCUMENT QUERIES (Hit-Rate & Similarity Scores)")
    print("-" * 75)

    hits = 0
    in_doc_scores = []

    for item in IN_DOCUMENT_QUERIES:
        query = item["query"]
        expected_page = item["expected_page"]

        results = vstore.search_with_similarity(query, top_k=4)
        top_score = results[0][1] if results else 0.0
        in_doc_scores.append(top_score)

        retrieved_pages = [int(doc.metadata.get("page", 0)) for doc, _ in results]
        hit = expected_page in retrieved_pages
        if hit:
            hits += 1

        status_icon = "PASS" if hit else "FAIL"
        print(f"[{status_icon}] Query: '{query[:48]}...'")
        print(f"       Expected Page: {expected_page} | Retrieved Pages: {retrieved_pages}")
        print(f"       Top Cosine Similarity: {top_score:.4f}\n")

    hit_rate = (hits / len(IN_DOCUMENT_QUERIES)) * 100
    min_in_doc_score = min(in_doc_scores)
    avg_in_doc_score = sum(in_doc_scores) / len(in_doc_scores)

    # Evaluate Off-Topic Queries
    print("-" * 75)
    print("  EVALUATING OFF-TOPIC QUERIES (Should Be Refused / Low Similarity)")
    print("-" * 75)

    off_topic_scores = []
    for query in OFF_TOPIC_QUERIES:
        results = vstore.search_with_similarity(query, top_k=4)
        top_score = results[0][1] if results else 0.0
        off_topic_scores.append(top_score)
        print(f"[OFF-TOPIC] Query: '{query}'")
        print(f"            Top Cosine Similarity: {top_score:.4f}\n")

    max_off_topic_score = max(off_topic_scores)
    avg_off_topic_score = sum(off_topic_scores) / len(off_topic_scores)

    # Threshold Recommendation Analysis
    # Ideal threshold lies halfway between max off-topic score and min in-doc score
    recommended_threshold = round((max_off_topic_score + min_in_doc_score) / 2.0, 2)

    print("=" * 75)
    print("  BENCHMARK SUMMARY & EMPIRICAL THRESHOLD CALIBRATION")
    print("=" * 75)
    print(f"Total In-Document Queries Evaluated: {len(IN_DOCUMENT_QUERIES)}")
    print(f"Retrieval Top-4 Hit Rate:             {hits}/{len(IN_DOCUMENT_QUERIES)} ({hit_rate:.1f}%)")
    print(f"In-Document Top Similarity:          Min={min_in_doc_score:.4f}, Avg={avg_in_doc_score:.4f}")
    print(f"Off-Topic Top Similarity:             Max={max_off_topic_score:.4f}, Avg={avg_off_topic_score:.4f}")
    print(f"Separation Margin (Min In - Max Off): {min_in_doc_score - max_off_topic_score:.4f}")
    print(f"\n>> EMPIRICALLY RECOMMENDED RELEVANCE_THRESHOLD: {recommended_threshold}")
    print("=" * 75)


if __name__ == "__main__":
    run_evaluation()
