"""
CSV utilities shared by the Flask server. Search itself now lives in
db.search() (Postgres + pgvector); run the Flask server for queries.
"""

from csv import DictReader


def get_data(path: str) -> list[dict]:
    with open(path, newline='', encoding='utf-8') as f:
        rows = list(DictReader(f))
        for row in rows:
            for key in row:
                if row[key]:
                    row[key] = row[key].strip()
        return rows


def build_embed_text(row: dict) -> str:
    """Concatenate the most semantically rich fields for embedding."""
    parts = [
        row.get('company_name', ''),
        row.get('short_description', ''),
        row.get('product_description', ''),
        row.get('mapped_function', ''),
        row.get('mapped_industry', ''),
        row.get('match_keywords', ''),
    ]
    return ' | '.join(p for p in parts if p.strip())
