from langchain_core.documents import Document
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_chroma import Chroma

from csv import DictReader

import chromadb


def get_data(path: str) -> list[dict]:
    with open(path, newline='', encoding='utf-8') as f:
        rows = list(DictReader(f))
        # Strip whitespace from all values in case CSV has extra spaces
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


def init_store(csv_path: str = 'test.csv') -> Chroma:
    embeddings = OpenAIEmbeddings(model='text-embedding-3-small')

    chroma_client = chromadb.PersistentClient(path='./omni.db')

    vector_store = Chroma(
        client=chroma_client,
        collection_name='companies',
        embedding_function=embeddings,
        collection_metadata={'hnsw:space': 'cosine'},
    )

    rows = get_data(csv_path)
    print(f'Loaded {len(rows)} total rows from CSV.')
    if rows:
        print(f'First row keys: {list(rows[0].keys())}')
        print(f'First row active value: {repr(rows[0].get("active"))}')
    
    active_rows = [r for r in rows if r.get('active', '').lower() == 'true']
    print(f'Found {len(active_rows)} active rows (active=true).')
    
    if len(active_rows) == 0 and len(rows) > 0:
        # Debug: Check what values are in the active column
        active_values = set(r.get('active', '') for r in rows)
        print(f'DEBUG: Unique values in "active" column: {active_values}')

    documents = []
    for row in active_rows:
        content = build_embed_text(row)
        if not content.strip():
            print(f'  Warning: Skipping row with empty content: {row.get("company_name", "UNKNOWN")}')
            continue
        doc = Document(
            page_content=content,
            metadata={
                'company_name':    row.get('company_name', ''),
                'website':         row.get('website', ''),
                'short_description': row.get('short_description', ''),
                'mapped_function': row.get('mapped_function', ''),
                'mapped_industry': row.get('mapped_industry', ''),
                'match_keywords':  row.get('match_keywords', ''),
                'aliases':         row.get('aliases', ''),
                'priority':        row.get('priority', ''),
            },
        )
        documents.append(doc)

    print(f'Loading {len(documents)} companies into ChromaDB...')
    if len(documents) == 0:
        print('Error: No documents to load. Check that CSV has rows with active=true and non-empty content.')
        return vector_store
    
    vector_store.add_documents(documents)
    print(f'Done. {len(documents)} documents stored with cosine similarity index.')

    return vector_store


def search(vector_store: Chroma, query: str, k: int = 5) -> list[tuple[Document, float]]:
    """Return top-k results with cosine similarity scores (higher = more similar)."""

    results = vector_store.similarity_search_with_relevance_scores(query, k=k)
    top_result_score = results[0]
    while top_result_score < 0.8:
        response = camille_input('> ')
        query += response
    return results


def print_results(results: list[tuple[Document, float]]) -> None:
    print(f'\n{"─" * 60}')
    for i, (doc, score) in enumerate(results, 1):
        m = doc.metadata
        print(f'#{i}  {m["company_name"]}  (score: {score:.4f})')
        print(f'    {m["short_description"]}')
        print(f'    Function: {m["mapped_function"]}  |  Industry: {m["mapped_industry"]}')
        print(f'    {m["website"]}')
        print()


def load_store() -> Chroma:
    """Load an already-populated store without re-ingesting."""
    embeddings = OpenAIEmbeddings(model='text-embedding-3-small')
    chroma_client = chromadb.PersistentClient(path='./omni.db')
    return Chroma(
        client=chroma_client,
        collection_name='companies',
        embedding_function=embeddings,
        collection_metadata={'hnsw:space': 'cosine'},
    )


if __name__ == '__main__':
    import sys
    import os

    # Build the DB if it doesn't exist yet, otherwise just load it
    db_exists = os.path.isdir('./omni.db')

    if not db_exists or '--rebuild' in sys.argv:
        vector_store = init_store()
    else:
        vector_store = load_store()
        count = vector_store._collection.count()
        print(f'Loaded existing store ({count} documents).')

    # Interactive search loop
    print('\nCosine similarity search ready. Type a query or "quit" to exit.')
    while True:
        try:
            query = input('\nQuery> ').strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not query or query.lower() in ('quit', 'exit', 'q'):
            break

        results = search(vector_store, query, k=5)
        print_results(results)
