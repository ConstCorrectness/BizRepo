# Project Overview: Omni-Collection Search Engine

Omni-Collection is a high-performance search engine designed for exploring a curated collection of companies. It leverages **Semantic Search (Cosine Similarity)** using OpenAI embeddings and ChromaDB to provide more relevant results than traditional keyword matching.

The application features a dual-persistence model:
- **Vector Store:** ChromaDB stores document embeddings for fast similarity search.
- **Relational Store:** PostgreSQL (production) or SQLite (local) persists manual company entries and metadata.
- **CSV Ingestion:** Supports bulk loading from CSV files (e.g., `test.csv`).

## Core Technologies
- **Backend:** Python 3.12, Flask, Flask-CORS
- **AI/LLM:** LangChain, OpenAI (`text-embedding-3-small`)
- **Databases:** ChromaDB (Vector), PostgreSQL/SQLite (Relational)
- **Deployment:** Gunicorn, Heroku-ready (`Procfile`)
- **Frontend:** HTML5, CSS3 (Vanilla), jQuery

---

## Building and Running

### Prerequisites
- Python 3.12+
- An OpenAI API Key (`OPENAI_API_KEY`)

### Setup
1. **Install Dependencies:**
   ```powershell
   pip install -r requirements.txt
   ```
2. **Environment Variables:**
   Create a `.env` file or set the following:
   - `OPENAI_API_KEY`: Your OpenAI key.
   - `DATABASE_URL`: (Optional) Postgres connection string for production. Defaults to SQLite locally.

### Running the Application
- **Start the Web Server:**
  ```powershell
  python server.py
  ```
  The interface will be available at `http://localhost:5050`.

- **Interactive CLI Search:**
  ```powershell
  python main.py
  ```
  Use `--rebuild` to wipe the vector store and re-index from `test.csv`.

---

## Key Files & Architecture

- **`server.py`**: The primary API entry point. Handles web routing, search requests, and company management.
- **`main.py`**: Contains core ingestion logic, embedding functions, and the interactive CLI search tool.
- **`db.py`**: Database abstraction layer providing a unified interface for SQLite and PostgreSQL.
- **`index.html`**: A comprehensive single-page frontend that supports CSV uploads, manual entries, and real-time search.
- **`test.csv`**: The default dataset template containing company names, descriptions, and metadata.
- **`requirements.txt`**: Project dependencies including `langchain-chroma`, `flask`, and `openai`.

---

## Development Conventions

- **Data Ingestion:** Logic for processing CSVs and generating embeddings is centralized in `main.py` to ensure consistency between CLI and Web versions.
- **Company IDs:** Document IDs in ChromaDB are deterministic slugs derived from company names (see `_company_id` in `server.py`).
- **Persistence:** Manual additions via the UI are first saved to the relational database and then immediately indexed into the vector store.
- **Search Logic:** The system uses a `CONFIDENCE_THRESHOLD` (default 0.35) to flag low-confidence results to the user.
