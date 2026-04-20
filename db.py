"""
Persistence for manually added companies.

Uses SQLite locally (file at ./companies.db) and Postgres on Heroku
(when DATABASE_URL is set). Same public interface either way:

    init_db()
    ensure_unique_index()
    get_additions() -> list[dict]
    upsert_company(row: dict) -> dict
"""

import os
import sqlite3


FIELDS = [
    "company_name", "website", "short_description", "product_description",
    "mapped_function", "mapped_industry", "match_keywords", "aliases",
    "active", "priority", "source",
]

IS_POSTGRES = bool(os.environ.get("DATABASE_URL"))
SQLITE_PATH = os.environ.get("SQLITE_PATH", "./companies.db")


if IS_POSTGRES:
    import psycopg2
    from psycopg2.extras import RealDictCursor


def _pg_conn():
    url = os.environ["DATABASE_URL"].replace("postgres://", "postgresql://", 1)
    return psycopg2.connect(url)


def _sqlite_conn():
    conn = sqlite3.connect(SQLITE_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def backend() -> str:
    return "postgres" if IS_POSTGRES else f"sqlite ({SQLITE_PATH})"


# ── init ──────────────────────────────────────────────────────────────

def init_db() -> None:
    if IS_POSTGRES:
        with _pg_conn() as conn, conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS companies (
                    id               SERIAL PRIMARY KEY,
                    company_name     TEXT NOT NULL,
                    website          TEXT DEFAULT '',
                    short_description  TEXT DEFAULT '',
                    product_description TEXT DEFAULT '',
                    mapped_function  TEXT DEFAULT '',
                    mapped_industry  TEXT DEFAULT '',
                    match_keywords   TEXT DEFAULT '',
                    aliases          TEXT DEFAULT '',
                    active           TEXT DEFAULT 'true',
                    priority         TEXT DEFAULT '5',
                    source           TEXT DEFAULT 'manual entry',
                    created_at       TIMESTAMPTZ DEFAULT NOW()
                )
            """)
            conn.commit()
        return

    with _sqlite_conn() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS companies (
                id                  INTEGER PRIMARY KEY AUTOINCREMENT,
                company_name        TEXT NOT NULL COLLATE NOCASE,
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
                created_at          TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.commit()


def ensure_unique_index() -> None:
    if IS_POSTGRES:
        with _pg_conn() as conn, conn.cursor() as cur:
            cur.execute("""
                CREATE UNIQUE INDEX IF NOT EXISTS companies_name_unique
                ON companies (lower(company_name))
            """)
            conn.commit()
        return

    with _sqlite_conn() as conn:
        # COLLATE NOCASE on the column already makes this case-insensitive.
        conn.execute("""
            CREATE UNIQUE INDEX IF NOT EXISTS companies_name_unique
            ON companies (company_name COLLATE NOCASE)
        """)
        conn.commit()


# ── reads ─────────────────────────────────────────────────────────────

def get_additions() -> list[dict]:
    """All active companies from the persistence store."""
    if IS_POSTGRES:
        with _pg_conn() as conn, conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                f"SELECT {', '.join(FIELDS)} FROM companies "
                "WHERE lower(active) = 'true' ORDER BY created_at"
            )
            return [dict(r) for r in cur.fetchall()]

    with _sqlite_conn() as conn:
        cur = conn.execute(
            f"SELECT {', '.join(FIELDS)} FROM companies "
            "WHERE lower(active) = 'true' ORDER BY created_at"
        )
        return [dict(r) for r in cur.fetchall()]


# ── writes ────────────────────────────────────────────────────────────

def upsert_company(row: dict) -> dict:
    """Insert or update by company_name (case-insensitive). Returns the stored row."""
    payload = {f: row.get(f, "") for f in FIELDS}

    if IS_POSTGRES:
        with _pg_conn() as conn, conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                f"""
                INSERT INTO companies ({', '.join(FIELDS)})
                VALUES ({', '.join(f'%({f})s' for f in FIELDS)})
                ON CONFLICT (lower(company_name))
                DO UPDATE SET
                    {', '.join(f'{f} = EXCLUDED.{f}' for f in FIELDS if f != 'company_name')},
                    created_at = NOW()
                RETURNING {', '.join(FIELDS)}
                """,
                payload,
            )
            result = dict(cur.fetchone())
            conn.commit()
        return result

    with _sqlite_conn() as conn:
        placeholders = ', '.join(f':{f}' for f in FIELDS)
        update_cols  = ', '.join(
            f'{f} = excluded.{f}' for f in FIELDS if f != 'company_name'
        )
        conn.execute(
            f"""
            INSERT INTO companies ({', '.join(FIELDS)})
            VALUES ({placeholders})
            ON CONFLICT(company_name) DO UPDATE SET
                {update_cols},
                created_at = CURRENT_TIMESTAMP
            """,
            payload,
        )
        conn.commit()
        cur = conn.execute(
            f"SELECT {', '.join(FIELDS)} FROM companies "
            "WHERE company_name = :company_name COLLATE NOCASE",
            {"company_name": payload["company_name"]},
        )
        return dict(cur.fetchone())
