"""
Flask API — ChromaDB cosine similarity search + serves index.html.

Local:  python server.py          → http://localhost:5050
Heroku: gunicorn server:app       → $PORT (set automatically)

Required env vars:
    OPENAI_API_KEY   — OpenAI key for embeddings
    DATABASE_URL     — set automatically by Heroku Postgres addon

Endpoints:
    GET  /                       → index.html
    GET  /health                 → { status, count, db }
    GET  /search?q=...&k=5       → { query, results, low_confidence }
    POST /companies              → add/update a company (persists to Postgres + indexes live)
    POST /rebuild                → wipe ChromaDB and re-index from test.csv + Postgres
"""

import os
import re
from flask import Flask, jsonify, request, send_file
from flask_cors import CORS

from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
import chromadb

from main import get_data, build_embed_text
import db


def _company_id(name: str) -> str:
    """Deterministic, stable document ID from company name (slug)."""
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")

app = Flask(__name__)
CORS(app)

CONFIDENCE_THRESHOLD = 0.35

_store: Chroma | None = None
_db_ready = False


# ── DB init ───────────────────────────────────────────────────────────

def ensure_db():
    global _db_ready
    if _db_ready:
        return
    if os.environ.get("DATABASE_URL"):
        try:
            db.init_db()
            db.ensure_unique_index()
            _db_ready = True
            print("[db] Postgres ready")
        except Exception as e:
            print(f"[db] WARNING: Postgres unavailable — {e}")
    else:
        print("[db] No DATABASE_URL — running without persistence")
        _db_ready = True


# ── vector store ──────────────────────────────────────────────────────

def build_store() -> Chroma:
    """
    Build a fresh EphemeralClient from:
      1. test.csv  (base dataset, committed to git)
      2. Postgres  (manually added companies, survives dyno restarts)
    """
    ensure_db()

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    client = chromadb.EphemeralClient()
    store = Chroma(
        client=client,
        collection_name="companies",
        embedding_function=embeddings,
        collection_metadata={"hnsw:space": "cosine"},
    )

    # Load base rows from CSV
    csv_rows = get_data("test.csv")
    csv_active = [r for r in csv_rows if r.get("active", "").lower() == "true"]

    # Load persisted additions from Postgres
    pg_rows: list[dict] = []
    if _db_ready and os.environ.get("DATABASE_URL"):
        try:
            pg_rows = db.get_additions()
        except Exception as e:
            print(f"[db] Could not load additions: {e}")

    # Merge — Postgres additions override CSV rows with same name
    csv_names = {r["company_name"].lower() for r in csv_active}
    merged = list(csv_active)
    for row in pg_rows:
        if row["company_name"].lower() not in csv_names:
            merged.append(row)
        # if it IS in CSV we still take the Postgres version (user edited it)
        else:
            merged = [row if r["company_name"].lower() == row["company_name"].lower()
                      else r for r in merged]

    docs = [
        Document(
            page_content=build_embed_text(r),
            metadata={k: r.get(k, "") for k in [
                "company_name", "website", "short_description",
                "mapped_function", "mapped_industry",
                "match_keywords", "aliases", "priority",
            ]},
        )
        for r in merged
        if build_embed_text(r).strip()
    ]

    ids = [_company_id(r["company_name"]) for r in merged if build_embed_text(r).strip()]

    print(f"[store] Indexing {len(docs)} companies "
          f"({len(csv_active)} CSV + {len(pg_rows)} from Postgres)…")
    store.add_documents(docs, ids=ids)
    print(f"[store] Done.")
    return store


def get_store() -> Chroma:
    global _store
    if _store is None:
        _store = build_store()
    return _store


def _doc_to_result(doc: Document, score: float) -> dict:
    m = doc.metadata
    return {
        "company_name":      m.get("company_name", ""),
        "website":           m.get("website", ""),
        "short_description": m.get("short_description", ""),
        "mapped_function":   m.get("mapped_function", ""),
        "mapped_industry":   m.get("mapped_industry", ""),
        "match_keywords":    m.get("match_keywords", ""),
        "aliases":           m.get("aliases", ""),
        "priority":          m.get("priority", ""),
        "score":             round(float(score), 4),
    }


# ── routes ────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return send_file("index.html")


@app.route("/health")
def health():
    try:
        count = get_store()._collection.count()
        return jsonify({
            "status": "ok",
            "count": count,
            "db": bool(os.environ.get("DATABASE_URL")),
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/search")
def search():
    query = request.args.get("q", "").strip()
    k = min(int(request.args.get("k", 5)), 20)
    if not query:
        return jsonify({"error": 'param "q" required'}), 400

    raw = get_store().similarity_search_with_relevance_scores(query, k=k)
    results = [_doc_to_result(doc, score) for doc, score in raw]
    top = results[0]["score"] if results else 0.0

    return jsonify({
        "query":          query,
        "results":        results,
        "low_confidence": top < CONFIDENCE_THRESHOLD,
        "threshold":      CONFIDENCE_THRESHOLD,
    })


@app.route("/companies", methods=["POST"])
def add_company():
    """
    Add or update a company.
    Persists to Postgres, then embeds + upserts into the live ChromaDB collection.
    """
    data = request.get_json(force=True)

    required = ["company_name", "short_description", "mapped_function", "mapped_industry"]
    missing = [f for f in required if not str(data.get(f, "")).strip()]
    if missing:
        return jsonify({"error": f"Missing required fields: {', '.join(missing)}"}), 400

    row = {
        "company_name":       str(data.get("company_name", "")).strip(),
        "website":            str(data.get("website", "")).strip(),
        "short_description":  str(data.get("short_description", "")).strip(),
        "product_description": str(data.get("product_description", "")).strip(),
        "mapped_function":    str(data.get("mapped_function", "")).strip(),
        "mapped_industry":    str(data.get("mapped_industry", "")).strip(),
        "match_keywords":     str(data.get("match_keywords", "")).strip(),
        "aliases":            str(data.get("aliases", "")).strip(),
        "active":             "true",
        "priority":           str(data.get("priority", "5")).strip(),
        "source":             str(data.get("source", "manual entry")).strip(),
    }

    # 1. Persist to Postgres
    if _db_ready and os.environ.get("DATABASE_URL"):
        try:
            row = db.upsert_company(row)
        except Exception as e:
            return jsonify({"error": f"DB error: {e}"}), 500

    # 2. Upsert into the live ChromaDB collection
    store = get_store()
    embed_text = build_embed_text(row)
    doc_id = _company_id(row["company_name"])

    # Delete by deterministic ID (reliable; where-filter delete is broken in some versions)
    try:
        store._collection.delete(ids=[doc_id])
    except Exception as e:
        print(f"[store] delete warning for '{doc_id}': {e}")

    doc = Document(
        page_content=embed_text,
        metadata={k: row.get(k, "") for k in [
            "company_name", "website", "short_description",
            "mapped_function", "mapped_industry",
            "match_keywords", "aliases", "priority",
        ]},
    )
    store.add_documents([doc], ids=[doc_id])

    return jsonify({
        "status":  "ok",
        "company": row["company_name"],
        "indexed": True,
    }), 201


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
    get_store()
    port = int(os.environ.get("PORT", 5050))
    app.run(host="0.0.0.0", port=port, debug=False)
