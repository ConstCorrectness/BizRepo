"""
Flask API — ChromaDB cosine similarity search + serves index.html.

Local:  python server.py          → http://localhost:5050
Heroku: gunicorn server:app       → $PORT (set automatically)

Config vars (set in Heroku dashboard or with `heroku config:set`):
    OPENAI_API_KEY=sk-...

Endpoints:
    GET  /              → index.html
    GET  /health        → { status, count }
    GET  /search?q=...&k=5 → { query, results, low_confidence }
    POST /rebuild       → wipe and re-index from test.csv
"""

import os
from flask import Flask, jsonify, request, send_file
from flask_cors import CORS

from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_chroma import Chroma
import chromadb

from main import get_data, build_embed_text
from langchain_core.documents import Document

app = Flask(__name__)
CORS(app)

CONFIDENCE_THRESHOLD = 0.35

# ── vector store singleton ────────────────────────────────────────────
_store: Chroma | None = None


def build_store() -> Chroma:
    """Always build fresh from test.csv (ephemeral — safe for Heroku)."""
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    # EphemeralClient: in-memory, no disk writes → works on Heroku
    client = chromadb.EphemeralClient()

    store = Chroma(
        client=client,
        collection_name="companies",
        embedding_function=embeddings,
        collection_metadata={"hnsw:space": "cosine"},
    )

    rows = get_data("test.csv")
    active = [r for r in rows if r.get("active", "").lower() == "true"]

    docs = [
        Document(
            page_content=build_embed_text(r),
            metadata={
                "company_name":      r.get("company_name", ""),
                "website":           r.get("website", ""),
                "short_description": r.get("short_description", ""),
                "mapped_function":   r.get("mapped_function", ""),
                "mapped_industry":   r.get("mapped_industry", ""),
                "match_keywords":    r.get("match_keywords", ""),
                "aliases":           r.get("aliases", ""),
                "priority":          r.get("priority", ""),
            },
        )
        for r in active
        if build_embed_text(r).strip()
    ]

    print(f"[store] Indexing {len(docs)} companies…")
    store.add_documents(docs)
    print(f"[store] Done — {len(docs)} docs in collection.")
    return store


def get_store() -> Chroma:
    global _store
    if _store is None:
        _store = build_store()
    return _store


# ── routes ────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return send_file("index.html")


@app.route("/health")
def health():
    try:
        count = get_store()._collection.count()
        return jsonify({"status": "ok", "count": count})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/search")
def search():
    query = request.args.get("q", "").strip()
    k = min(int(request.args.get("k", 5)), 20)

    if not query:
        return jsonify({"error": 'query param "q" is required'}), 400

    raw = get_store().similarity_search_with_relevance_scores(query, k=k)

    results = [
        {
            "company_name":      doc.metadata.get("company_name", ""),
            "website":           doc.metadata.get("website", ""),
            "short_description": doc.metadata.get("short_description", ""),
            "mapped_function":   doc.metadata.get("mapped_function", ""),
            "mapped_industry":   doc.metadata.get("mapped_industry", ""),
            "match_keywords":    doc.metadata.get("match_keywords", ""),
            "aliases":           doc.metadata.get("aliases", ""),
            "priority":          doc.metadata.get("priority", ""),
            "score":             round(float(score), 4),
        }
        for doc, score in raw
    ]

    top_score = results[0]["score"] if results else 0.0

    return jsonify({
        "query":          query,
        "results":        results,
        "low_confidence": top_score < CONFIDENCE_THRESHOLD,
        "threshold":      CONFIDENCE_THRESHOLD,
    })


@app.route("/rebuild", methods=["POST"])
def rebuild():
    global _store
    _store = None
    try:
        _store = build_store()
        return jsonify({"status": "rebuilt", "count": _store._collection.count()})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


# ── startup ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    get_store()  # eager init so first request is fast
    port = int(os.environ.get("PORT", 5050))
    app.run(host="0.0.0.0", port=port, debug=False)
