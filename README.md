# PDF RAG Assistant

A simple **Retrieval-Augmented Generation (RAG)** project that allows users to ask questions about a PDF from the terminal.

## Tech Stack

* Python
* LangChain
* PyPDF
* Hugging Face Embeddings
* ChromaDB
* OpenRouter LLM API

## RAG Flow

```text
PDF
 ↓
Text Extraction
 ↓
Chunking
 ↓
Embeddings
 ↓
ChromaDB
 ↓
Similarity Search
 ↓
Relevant Chunks
 ↓
LLM
 ↓
Answer
```

## Setup

### 1. Clone the repository

```bash
git clone <repository-url>
cd RAG_Project
```

### 2. Create virtual environment

```bash
python -m venv .venv
```

Windows:

```powershell
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Add API key

Create a `.env` file:

```env
OPENROUTER_API_KEY=your_api_key_here
```

### 5. Add your PDF

Place your PDF in the project directory and update:

```python
PDF_PATH = "sample.pdf"
```

### 6. Run

```bash
python rag.py
```

Then ask questions directly in the terminal.

```text
You: Is cyber liability included in the policy?

Assistant: ...
```

Type `exit` to stop.

## How It Works

The PDF is split into smaller chunks and converted into embeddings using a local Hugging Face model. ChromaDB stores these embeddings and retrieves the most relevant chunks when a question is asked. The retrieved context is then sent to an LLM through OpenRouter to generate the final answer.

## Project Structure

```text
RAG_Project/
├── rag.py
├── requirements.txt
├── README.md
├── .env
└── sample.pdf
```

