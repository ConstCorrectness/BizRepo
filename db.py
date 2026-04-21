"""
Single-store persistence: Postgres + pgvector.

All company data AND embeddings live in one `companies` table. Search is an
SQL query against the hnsw-indexed `embedding` column.

Local dev: run `docker compose up -d` and `DATABASE_URL=postgresql://postgres:postgres@127.0.0.1:5434/omni`.
Heroku:    DATABASE_URL is set automatically by the Postgres addon.

Public interface:
    init_db()                               — extension, table, index
    is_empty() -> bool                      — true if no active rows
    get_all() -> list[dict]                 — all active rows (no embedding)
    upsert_company(row, embedding) -> dict  — single write, includes embedding
    search(query_embedding, k) -> list[(dict, score)]
    backend() -> str
"""

import os

import psycopg2
from psycopg2.extras import RealDictCursor, execute_values
from pgvector.psycopg2 import register_vector


EMBED_DIM = 1536  # text-embedding-3-small

FIELDS = [
    "company_name", "website", "short_description", "product_description",
    "mapped_function", "mapped_industry", "match_keywords", "aliases",
    "active", "priority", "source",
]


def _raw_conn():
    url = os.environ.get("DATABASE_URL")
    if not url:
        raise RuntimeError(
            "DATABASE_URL is not set. Run `docker compose up -d` and export "
            "DATABASE_URL=postgresql://postgres:postgres@127.0.0.1:5434/omni"
        )
    # Heroku's DATABASE_URL uses the legacy postgres:// scheme
    url = url.replace("postgres://", "postgresql://", 1)
    return psycopg2.connect(url)


def _conn():
    """Connection with pgvector adapters registered. Requires the extension to exist."""
    conn = _raw_conn()
    register_vector(conn)
    return conn


def backend() -> str:
    return "postgres+pgvector"


# ── init ──────────────────────────────────────────────────────────────

def init_db() -> None:
    # Create extension on a raw connection — pgvector adapters need the type
    # to already exist before register_vector() runs.
    with _raw_conn() as conn, conn.cursor() as cur:
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
        conn.commit()

    with _conn() as conn, conn.cursor() as cur:
        cur.execute(f"""
            CREATE TABLE IF NOT EXISTS companies (
                id                  SERIAL PRIMARY KEY,
                company_name        TEXT NOT NULL,
                website             TEXT DEFAULT '',
                short_description   TEXT DEFAULT '',
                product_description TEXT DEFAULT '',
                mapped_function     TEXT DEFAULT '',
                mapped_industry     TEXT DEFAULT '',
                match_keywords      TEXT DEFAULT '',
                aliases             TEXT DEFAULT '',
                active              TEXT DEFAULT 'true',
                priority            TEXT DEFAULT '5',
                source              TEXT DEFAULT 'manual entry',
                embedding           vector({EMBED_DIM}),
                created_at          TIMESTAMPTZ DEFAULT NOW()
            )
        """)
        cur.execute("""
            CREATE UNIQUE INDEX IF NOT EXISTS companies_name_unique
            ON companies (lower(company_name))
        """)
        cur.execute("""
            CREATE INDEX IF NOT EXISTS companies_embedding_hnsw
            ON companies USING hnsw (embedding vector_cosine_ops)
        """)
        conn.commit()


# ── reads ─────────────────────────────────────────────────────────────

def is_empty() -> bool:
    with _conn() as conn, conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM companies WHERE lower(active) = 'true'")
        return cur.fetchone()[0] == 0


def count_active() -> int:
    with _conn() as conn, conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM companies WHERE lower(active) = 'true'")
        return cur.fetchone()[0]


def get_all() -> list[dict]:
    """All active companies (without the embedding column)."""
    with _conn() as conn, conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute(
            f"SELECT {', '.join(FIELDS)} FROM companies "
            "WHERE lower(active) = 'true' ORDER BY created_at"
        )
        return [dict(r) for r in cur.fetchall()]


def search(query_embedding: list[float], k: int = 5) -> list[tuple[dict, float]]:
    """
    Cosine similarity search. Returns (row_dict, score) where score = 1 - distance
    (higher is more similar, matches the old Chroma relevance-score semantics).
    """
    with _conn() as conn, conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute(
            f"""
            SELECT {', '.join(FIELDS)},
                   1 - (embedding <=> %s::vector) AS score
            FROM companies
            WHERE lower(active) = 'true' AND embedding IS NOT NULL
            ORDER BY embedding <=> %s::vector
            LIMIT %s
            """,
            (query_embedding, query_embedding, k),
        )
        return [(dict(r), float(r["score"])) for r in cur.fetchall()]


# ── writes ────────────────────────────────────────────────────────────

def upsert_company(row: dict, embedding: list[float]) -> dict:
    """Insert or update by company_name (case-insensitive). Single atomic write."""
    payload = {f: row.get(f, "") for f in FIELDS}
    payload["embedding"] = embedding

    cols = FIELDS + ["embedding"]
    updates = ', '.join(f'{c} = EXCLUDED.{c}' for c in cols if c != 'company_name')

    with _conn() as conn, conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute(
            f"""
            INSERT INTO companies ({', '.join(cols)})
            VALUES ({', '.join(f'%({c})s' for c in cols)})
            ON CONFLICT (lower(company_name))
            DO UPDATE SET
                {updates},
                created_at = NOW()
            RETURNING {', '.join(FIELDS)}
            """,
            payload,
        )
        result = dict(cur.fetchone())
        conn.commit()
    return result


def upsert_many(rows_with_embeddings: list[tuple[dict, list[float]]]) -> int:
    """Bulk upsert. Used for CSV bootstrap and /rebuild."""
    if not rows_with_embeddings:
        return 0

    cols = FIELDS + ["embedding"]
    updates = ', '.join(f'{c} = EXCLUDED.{c}' for c in cols if c != 'company_name')

    values = [
        tuple(row.get(f, "") for f in FIELDS) + (embedding,)
        for row, embedding in rows_with_embeddings
    ]

    with _conn() as conn, conn.cursor() as cur:
        execute_values(
            cur,
            f"""
            INSERT INTO companies ({', '.join(cols)})
            VALUES %s
            ON CONFLICT (lower(company_name))
            DO UPDATE SET
                {updates},
                created_at = NOW()
            """,
            values,
        )
        conn.commit()
    return len(values)
