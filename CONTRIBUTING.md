# Contributing to PrecisionRAG

Thank you for your interest in contributing! This guide will help you get started.

## Setup

1. **Clone the repo**
   `ash
   git clone https://github.com/Anantam-Singh/Vector_Database_Design_for_RAG_System.git
   cd Vector_Database_Design_for_RAG_System
   `

2. **Create a virtual environment**
   `ash
   uv venv .venv --python 3.11
   uv pip install -r requirements.lock.txt --extra-index-url https://download.pytorch.org/whl/cu128 --index-strategy unsafe-best-match
   `

3. **Set up environment variables**
   `ash
   copy .env.example .env
   # Add your GROQ_API_KEY in .env (only needed for RAGAS evaluation)
   `

4. **Start Qdrant**
   `ash
   start_qdrant.bat
   `

5. **Verify Qdrant is reachable** (before running the full index build)
   `ash
   .venv\Scripts\python -c "from qdrant_client import QdrantClient; c = QdrantClient('http://127.0.0.1:6333'); print('Qdrant OK:', c.get_collections())"
   `

6. **Build the index**
   `ash
   .venv\Scripts\python -m scripts.build_index --size 100000
   `

## Running Tests

`ash
.venv\Scripts\python -m pytest tests/
`

## Making Changes

- Create a new branch: git checkout -b your-branch-name
- Make your changes
- Run tests to make sure nothing is broken
- Open a Pull Request against main

## Code Style

- Python 3.11+
- Follow existing code conventions in the precisionrag/ module
- Keep all tunable parameters in config.yaml (not hardcoded)

## Notes

- Windows users: always use 127.0.0.1 instead of localhost for Qdrant URL to avoid IPv6 fallback delays
- GPU is optional — the system auto-detects and switches between CPU and GPU profiles

## Questions?

Open an issue and we'll be happy to help!
