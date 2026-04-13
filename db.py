"""
Postgres helpers for persisting manually added companies.

DATABASE_URL is set automatically by Heroku when you provision
the heroku-postgresql addon.
"""

import os
import psycopg2
from psycopg2.extras import RealDictCursor


FIELDS = [
    "company_name", "website", "short_description", "product_description",
    "mapped_function", "mapped_industry", "match_keywords", "aliases",
    "active", "priority", "source",
]


def _conn():
    url = os.environ.get("DATABASE_URL", "")
    # Heroku gives postgres:// but psycopg2 needs postgresql://
    url = url.replace("postgres://", "postgresql://", 1)
    return psycopg2.connect(url)


def init_db() -> None:
    """Create the companies table if it doesn't exist yet."""
    with _conn() as conn, conn.cursor() as cur:
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


def get_additions() -> list[dict]:
    """Return all manually added companies (active=true rows from Postgres)."""
    with _conn() as conn, conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute(
            f"SELECT {', '.join(FIELDS)} FROM companies "
            "WHERE lower(active) = 'true' ORDER BY created_at"
        )
        return [dict(r) for r in cur.fetchall()]


def upsert_company(row: dict) -> dict:
    """
    Insert or update a company by company_name (case-insensitive).
    Returns the final stored row.
    """
    with _conn() as conn, conn.cursor(cursor_factory=RealDictCursor) as cur:
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
            {f: row.get(f, "") for f in FIELDS},
        )
        result = dict(cur.fetchone())
        conn.commit()
    return result


def ensure_unique_index() -> None:
    """Add a unique index on lower(company_name) for upsert support."""
    with _conn() as conn, conn.cursor() as cur:
        cur.execute("""
            CREATE UNIQUE INDEX IF NOT EXISTS companies_name_unique
            ON companies (lower(company_name))
        """)
        conn.commit()
