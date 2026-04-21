"""
Flask API — Postgres + pgvector cosine similarity search. Serves index.html.

Local:  docker compose up -d && python server.py   → http://localhost:5050
Heroku: gunicorn server:app                         → $PORT

Required env vars:
    OPENAI_API_KEY   — OpenAI key for embeddings
    DATABASE_URL     — Postgres connection string (set automatically on Heroku)

Endpoints:
    GET  /                 → index.html
    GET  /health           → { status, count, db }
    GET  /search?q=...&k=5 → { query, results, low_confidence }
    POST /companies        → add/update a company (single write to Postgres)
    POST /rebuild          → re-embed every row (e.g. after an embedding-model change)
"""

import os

from flask import Flask, jsonify, request, send_file
from flask_cors import CORS

from langchain_openai.embeddings import OpenAIEmbeddings

from main import get_data, build_embed_text
import db


app = Flask(__name__)
CORS(app)

CONFIDENCE_THRESHOLD = 0.35

_embeddings: OpenAIEmbeddings | None = None
_ready = False


# ── embeddings ────────────────────────────────────────────────────────

def get_embeddings() -> OpenAIEmbeddings:
    global _embeddings
    if _embeddings is None:
        _embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    return _embeddings


def embed_one(text: str) -> list[float]:
    return get_embeddings().embed_query(text)


def embed_many(texts: list[str]) -> list[list[float]]:
    return get_embeddings().embed_documents(texts)


# ── startup: init schema + one-time CSV bootstrap ─────────────────────

def ensure_ready() -> None:
    global _ready
    if _ready:
        return

    db.init_db()

    if db.is_empty():
        _bootstrap_from_csv("test.csv")
    else:
        _backfill_missing_embeddings()

    _ready = True
    print(f"[store] ready — {db.count_active()} companies indexed")


def _backfill_missing_embeddings() -> None:
    """Embed any rows that were persisted before the embedding column existed."""
    rows = db.rows_missing_embedding()
    if not rows:
        return
    print(f"[migrate] embedding {len(rows)} rows that have no vector…")
    texts = [build_embed_text(r) for r in rows]
    vectors = embed_many(texts)
    db.upsert_many(list(zip(rows, vectors)))
    print(f"[migrate] done.")


def _bootstrap_from_csv(path: str) -> None:
    rows = get_data(path)
    active = [
        r for r in rows
        if r.get("active", "").lower() == "true" and build_embed_text(r).strip()
    ]
    if not active:
        print(f"[bootstrap] no active rows in {path} — skipping")
        return

    print(f"[bootstrap] embedding {len(active)} companies from {path}…")
    texts = [build_embed_text(r) for r in active]
    vectors = embed_many(texts)

    db.upsert_many(list(zip(active, vectors)))
    print(f"[bootstrap] done.")


# ── routes ────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return send_file("index.html")


@app.route("/health")
def health():
    try:
        ensure_ready()
        return jsonify({
            "status":     "ok",
            "count":      db.count_active(),
            "db":         True,
            "db_backend": db.backend(),
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/search")
def search():
    query = request.args.get("q", "").strip()
    k = min(int(request.args.get("k", 5)), 20)
    if not query:
        return jsonify({"error": 'param "q" required'}), 400

    ensure_ready()
    query_vec = embed_one(query)
    raw = db.search(query_vec, k=k)

    results = [
        {
            "company_name":      row.get("company_name", ""),
            "website":           row.get("website", ""),
            "short_description": row.get("short_description", ""),
            "mapped_function":   row.get("mapped_function", ""),
            "mapped_industry":   row.get("mapped_industry", ""),
            "match_keywords":    row.get("match_keywords", ""),
            "aliases":           row.get("aliases", ""),
            "priority":          row.get("priority", ""),
            "score":             round(score, 4),
        }
        for row, score in raw
    ]
    top = results[0]["score"] if results else 0.0

    return jsonify({
        "query":          query,
        "results":        results,
        "low_confidence": top < CONFIDENCE_THRESHOLD,
        "threshold":      CONFIDENCE_THRESHOLD,
    })


@app.route("/companies", methods=["POST"])
def add_company():
    """Add or update a company. One atomic write to Postgres, embedding included."""
    data = request.get_json(force=True)

    required = ["company_name", "short_description", "mapped_function", "mapped_industry"]
    missing = [f for f in required if not str(data.get(f, "")).strip()]
    if missing:
        return jsonify({"error": f"Missing required fields: {', '.join(missing)}"}), 400

    row = {
        "company_name":        str(data.get("company_name", "")).strip(),
        "website":             str(data.get("website", "")).strip(),
        "short_description":   str(data.get("short_description", "")).strip(),
        "product_description": str(data.get("product_description", "")).strip(),
        "mapped_function":     str(data.get("mapped_function", "")).strip(),
        "mapped_industry":     str(data.get("mapped_industry", "")).strip(),
        "match_keywords":      str(data.get("match_keywords", "")).strip(),
        "aliases":             str(data.get("aliases", "")).strip(),
        "active":              "true",
        "priority":            str(data.get("priority", "5")).strip(),
        "source":              str(data.get("source", "manual entry")).strip(),
    }

    ensure_ready()
    embedding = embed_one(build_embed_text(row))

    try:
        stored = db.upsert_company(row, embedding)
    except Exception as e:
        return jsonify({"error": f"DB error: {e}"}), 500

    return jsonify({
        "status":  "ok",
        "company": stored["company_name"],
        "indexed": True,
    }), 201


@app.route("/upload_csv", methods=["POST"])
def upload_csv():
    """Bulk-merge companies from an uploaded CSV."""
    if "file" not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files["file"]
    if not file or not file.filename:
        return jsonify({"error": "No selected file"}), 400

    try:
        import io
        import csv
        stream = io.StringIO(file.stream.read().decode("utf-8-sig"), newline="")
        reader = csv.DictReader(stream)
        
        # Robust header mapping: handle spaces and casing
        raw_rows = list(reader)
        if not raw_rows:
            return jsonify({"status": "ok", "count": 0, "message": "Empty CSV"})

        processed = []
        for raw in raw_rows:
            # Normalize keys
            row = { k.strip().lower().replace(" ", "_"): v for k, v in raw.items() if k }
            
            # Map common variations to our internal FIELDS
            clean = {
                "company_name":        row.get("company_name") or row.get("company") or "",
                "website":             row.get("website") or row.get("url") or "",
                "short_description":   row.get("short_description") or row.get("description") or "",
                "product_description": row.get("product_description") or "",
                "mapped_function":     row.get("mapped_function") or row.get("function") or "",
                "mapped_industry":     row.get("mapped_industry") or row.get("industry") or "",
                "match_keywords":      row.get("match_keywords") or row.get("keywords") or "",
                "aliases":             row.get("aliases") or "",
                "active":              row.get("active") or "true",
                "priority":            row.get("priority") or "5",
                "source":              row.get("source") or file.filename,
            }
            
            # Basic validation
            if not clean["company_name"] or not clean["short_description"]:
                continue
            
            # Normalize active
            if str(clean["active"]).lower() in ["no", "false", "0"]:
                clean["active"] = "false"
            else:
                clean["active"] = "true"
                
            processed.append(clean)

        if not processed:
            return jsonify({"error": "No valid rows found in CSV (need company_name and short_description)"}), 400

        ensure_ready()
        print(f"[upload] embedding {len(processed)} rows from {file.filename}…")
        texts = [build_embed_text(r) for r in processed]
        vectors = embed_many(texts)
        
        count = db.upsert_many(list(zip(processed, vectors)))
        return jsonify({
            "status": "ok",
            "count": count,
            "message": f"Successfully merged {count} companies."
        })

    except Exception as e:
        return jsonify({"error": f"Upload failed: {e}"}), 500


@app.route("/upload_batch", methods=["POST"])
def upload_batch():
    """Embed and upsert a batch of companies sent as JSON."""
    data = request.get_json(force=True)
    if not isinstance(data, list):
        return jsonify({"error": "Expected a list of companies"}), 400

    if not data:
        return jsonify({"status": "ok", "count": 0})

    processed = []
    for raw in data:
        # Normalize keys and map to FIELDS
        row = { k.strip().lower().replace(" ", "_"): v for k, v in raw.items() if k }
        clean = {
            "company_name":        row.get("company_name") or row.get("company") or "",
            "website":             row.get("website") or row.get("url") or "",
            "short_description":   row.get("short_description") or row.get("description") or "",
            "product_description": row.get("product_description") or "",
            "mapped_function":     row.get("mapped_function") or row.get("function") or "",
            "mapped_industry":     row.get("mapped_industry") or row.get("industry") or "",
            "match_keywords":      row.get("match_keywords") or row.get("keywords") or "",
            "aliases":             row.get("aliases") or "",
            "active":              row.get("active") or "true",
            "priority":            row.get("priority") or "5",
            "source":              row.get("source") or "batch upload",
        }
        
        if not clean["company_name"] or not clean["short_description"]:
            continue
            
        clean["active"] = "false" if str(clean["active"]).lower() in ["no", "false", "0"] else "true"
        processed.append(clean)

    if not processed:
        return jsonify({"status": "ok", "count": 0})

    ensure_ready()
    texts = [build_embed_text(r) for r in processed]
    vectors = embed_many(texts)
    
    count = db.upsert_many(list(zip(processed, vectors)))
    return jsonify({"status": "ok", "count": count})


@app.route("/rebuild", methods=["POST"])
def rebuild():
    """Re-embed every active row. Use after changing the embedding model."""
    try:
        ensure_ready()
        rows = db.get_all()
        if not rows:
            return jsonify({"status": "rebuilt", "count": 0})

        texts = [build_embed_text(r) for r in rows]
        vectors = embed_many(texts)
        db.upsert_many(list(zip(rows, vectors)))

        return jsonify({"status": "rebuilt", "count": len(rows)})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


# ── startup ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    ensure_ready()
    port = int(os.environ.get("PORT", 5050))
    app.run(host="0.0.0.0", port=port, debug=False)
