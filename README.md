# AI-Engineering-Topics

A collection of small, self-contained Python scripts and demos covering core AI engineering concepts — from transformer basics to prompt engineering, conversation memory, logging/observability for LLM apps, and Retrieval-Augmented Generation (RAG). Each folder is an independent topic you can run on its own.

## Project Setup

### Prerequisites

- Python 3.10+
- An [OpenAI API key](https://platform.openai.com/api-keys) (required for every script that calls `OpenAI()` — i.e. everything except the `intro/` examples)

### 1. Clone the repository

**Bash (Linux/macOS)**
```bash
git clone https://github.com/janinirvit/AI-Engineering-Topics.git
cd AI-Engineering-Topics
```

**PowerShell (Windows)**
```powershell
git clone https://github.com/janinirvit/AI-Engineering-Topics.git
cd AI-Engineering-Topics
```

### 2. Create and activate a virtual environment

**Bash (Linux/macOS)**
```bash
python3 -m venv venv
source venv/bin/activate
```

**PowerShell (Windows)**
```powershell
python -m venv venv
venv\Scripts\Activate.ps1
```

### 3. Install dependencies

**Bash (Linux/macOS)**
```bash
pip install -r requirements.txt
```

**PowerShell (Windows)**
```powershell
pip install -r requirements.txt
```

The `requirements.txt` includes:

| Package | Used for |
|---|---|
| `langchain` | Building LLM apps via the LangChain framework (`langchain_fundamentals/`) |
| `openai` | Calling the OpenAI Chat Completions API |
| `python-dotenv` | Loading environment variables (e.g. `OPENAI_API_KEY`) from a `.env` file |
| `transformers`, `torch` | Running local Hugging Face transformer models (`intro/`) |
| `chromadb` | Local vector database used for embeddings/retrieval (`rag_fundamentals/`) |
| `pandas` | Reading/handling tabular data such as `space_fact.csv` |
| `streamlit` | Web UI for the interactive RAG demo |
| `pypdf2` | Extracting text from PDF files for RAG over documents |

### 4. Configure environment variables

Copy the provided `.env.example` file to `.env` (this file is git-ignored) and fill in your actual values:

**Bash (Linux/macOS)**
```bash
cp .env.example .env
```

**PowerShell (Windows)**
```powershell
Copy-Item .env.example .env
```

Then edit `.env` and set your OpenAI credentials:

```
OPENAI_API_KEY=sk-your-key-here
```

Some RAG scripts also let you select a Nomic embedding endpoint, which is configured via:

```
NOMIC_API_BASE=https://your-nomic-endpoint
```

### 5. Run a script

Each folder is independent. Run any script directly with Python, e.g.:

**Bash (Linux/macOS)**
```bash
python3 openai-intro/openai-simple.py
```

**PowerShell (Windows)**
```powershell
python openai-intro\openai-simple.py
```

Streamlit-based demos are launched differently:

**Bash (Linux/macOS)**
```bash
streamlit run rag_fundamentals/rag_streamlit.py
```

**PowerShell (Windows)**
```powershell
streamlit run rag_fundamentals\rag_streamlit.py
```

The PDF-based RAG app follows the same pattern:

**Bash (Linux/macOS)**
```bash
streamlit run rag_fundamentals/rag_pdf_simple.py
```

**PowerShell (Windows)**
```powershell
streamlit run rag_fundamentals\rag_pdf_simple.py
```

## Folder Structure & Descriptions

### `intro/`
Beginner-friendly introduction to how transformer-based language models work, using Hugging Face `transformers` and the local `distilgpt2` model (no API key needed).
- **`intro_transformer.py`** — Minimal example: loads a `text-generation` pipeline and generates text from a single prompt.
- **`intro_transformer_enhanced.py`** — Expanded version with a menu to (1) run a batch demo over several prompts, (2) run an interactive prompt loop, or (3) explain the tokenization process (text → tokens → model → text) with a worked example.

### `openai-intro/`
Introductory examples of calling the OpenAI Chat Completions API directly.
- **`openai-simple.py`** — The simplest possible OpenAI API call: sends a system + user message and prints the response.
- **`prompts.py`** — A tour of prompting techniques, including few-shot prompting, direct/zero-shot prompting, chain-of-thought prompting, instruction prompting, role-play prompting, open-ended prompting, temperature/top-p sampling, and streaming responses.

### `langchain_fundamentals/`
Introductory example of using the LangChain framework instead of calling the OpenAI SDK directly.
- **`intro_lang.py`** — Minimal LangChain example: initializes a chat model via `init_chat_model("gpt-4o-mini", model_provider="openai")` and sends a single hard-coded prompt, printing the model's response.
- **`text_loaders.py`** — End-to-end RAG example over a plain-text file: loads `doc/dream.txt` with `TextLoader`, splits it into chunks with `RecursiveCharacterTextSplitter`, cleans/normalizes the chunk text, embeds the chunks with `OpenAIEmbeddings`, and indexes them in an in-memory FAISS vector store. Retrieves the top matching chunks for a query, then feeds them into a `ChatPromptTemplate` → `ChatOpenAI` → `StrOutputParser` chain to generate a context-grounded answer (e.g. a bullet-point summary of the speech).
- **`chatbot_qa.py`** — Web scraping RAG pipeline: uses `SeleniumURLLoader` to scrape content from multiple URLs, splits the documents into chunks with `RecursiveCharacterTextSplitter`, embeds them using `OpenAIEmbeddings`, and stores them in a ChromaDB vector store. Provides an interactive Q&A interface where you ask questions and the chatbot retrieves relevant context from the scraped documents to generate accurate, polite, and helpful responses with cited sources.

### `context-memory/`
Demonstrates how to give a chatbot conversational memory (vs. a stateless chatbot) and persist chat history to disk.
- **`ctx-memory.py`** — A chatbot that keeps full message history across turns, supports `save`/`load`/`summary` commands, and auto-summarizes older history once the conversation grows long.
- **`no-memory.py`** — A baseline chatbot that treats every message independently with no memory of prior turns, for comparison.
- **`conversation.json`** — Example saved conversation produced by `ctx-memory.py`.

### `logging-llm/`
Shows how to add structured logging/observability around LLM calls (useful for debugging and monitoring production chatbots).
- **`chatbot_logging.py`** — A `ChatBot` class that logs every user input and model response as structured JSON (timestamp, session ID, model name, response time, token usage) to both a log file and the console, and logs errors when API calls fail.
- **`chatbot_logs.json`** — Example JSON-lines log output produced by `chatbot_logging.py`.

### `rag_fundamentals/`
End-to-end examples of Retrieval-Augmented Generation (RAG): embedding documents, storing them in a vector database (ChromaDB), retrieving relevant context for a query, and augmenting an LLM prompt with that context.
- **`simple_rag.py`** — A CLI RAG pipeline over a small CSV of space facts (auto-generated into `space_fact.csv`). Lets you pick an LLM (OpenAI/Ollama) and embedding model (OpenAI/Chroma/Nomic), builds a ChromaDB collection, and answers sample queries using only retrieved context.
- **`rag_streamlit.py`** — The same space-facts RAG pipeline wrapped in an interactive Streamlit web app, with model-selection controls in the sidebar, a query box, and a "technical details" panel showing the augmented prompt and retrieved references.
- **`rag_pdf_simple.py`** — Extends the RAG pipeline to real documents: a `SimplePDFProcessor` reads a PDF, splits it into overlapping text chunks, embeds them, and retrieves relevant chunks to answer questions about the PDF's content. Upload your own PDF via the Streamlit file uploader, or use the bundled **`DeepSeek_R1.pdf`** as a sample document.
- **`chromadb/`** — Persistent ChromaDB vector store created automatically the first time you run `rag_pdf_simple.py` (or any script that uses `chromadb.PersistentClient`). Safe to delete if you want to reset stored embeddings/collections.

### Root files
- **`requirements.txt`** — Python dependencies for all scripts in the repo.
- **`requirements-dev.txt`** — Development-only tooling (currently [`ruff`](https://docs.astral.sh/ruff/), used for formatting/linting).
- **`pyproject.toml`** — Ruff configuration (line length, target Python version, enabled lint rules, formatting style).
- **`.env.example`** — Template for the environment variables used across the scripts (`OPENAI_API_KEY`, `NOMIC_API_BASE`). Copy it to `.env` and fill in your real values.
- **`space_fact.csv`** — Sample dataset of interesting space facts, auto-generated and consumed by the `rag_fundamentals/` scripts.

## Code Formatting & Linting

This project uses [Ruff](https://docs.astral.sh/ruff/) to format and lint all Python code.

Install the dev dependency:

**Bash (Linux/macOS)**
```bash
pip install -r requirements-dev.txt
```

**PowerShell (Windows)**
```powershell
pip install -r requirements-dev.txt
```

Format all files:

**Bash (Linux/macOS)**
```bash
ruff format .
```

**PowerShell (Windows)**
```powershell
ruff format .
```

Check for lint issues:

**Bash (Linux/macOS)**
```bash
ruff check .
```

**PowerShell (Windows)**
```powershell
ruff check .
```

Auto-fix safe lint issues:

**Bash (Linux/macOS)**
```bash
ruff check . --fix
```

**PowerShell (Windows)**
```powershell
ruff check . --fix
```

## Notes

- Scripts that call `OpenAI()` require `OPENAI_API_KEY` to be set (via `.env` or your environment) or they will raise/return an API error.
- The `intro/` scripts download the `distilgpt2` model from Hugging Face on first run and work fully offline afterward (no OpenAI key required).
- Log and conversation JSON files (`chatbot_logs.json`, `conversation.json`, `space_fact.csv`) are example outputs checked into the repo for reference — running the scripts again will regenerate/append to them.
