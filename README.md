# NPDE TA RAG System

A Retrieval-Augmented Generation (RAG) system for the ETH Zurich course **"Numerical Methods for Partial Differential Equations"** (401-0674-00L).

This tool helps Teaching Assistants quickly search course materials, find similar exam problems, and answer student questions using GPT-4o with context from course documents.

## Features

- **Semantic Search**: Find relevant content across lectures, homework, and past exams
- **AI-Powered Q&A**: Get accurate answers based on course materials using GPT-4o
- **Document Filtering**: Filter by document type (lecture/homework/exam) or exam year
- **Interactive Chat**: Multi-turn conversation with context awareness

## Prerequisites

- Python 3.11+
- OpenAI API key ([Get one here](https://platform.openai.com/api-keys))

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/Shizheng-Wen/NPDE_TA_RAG.git
cd NPDE_TA_RAG
```

### 2. Create a virtual environment

Using conda:
```bash
conda create -n npde_rag python=3.11 -y
conda activate npde_rag
```

Or using venv:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Set up your API key

Create a `.env` file in the project root:

```bash
echo "OPENAI_API_KEY=your_api_key_here" > .env
```

### 5. Add course materials

Create the data directories and add your PDF files:

```
data/
├── lectures/      # Course lecture notes (e.g., NUMPDE.pdf)
├── homework/      # Problem sets and projects
└── exams/         # Past exam papers with solutions
```

### 6. Index the documents

This step parses PDFs, generates embeddings, and stores them in the vector database:

```bash
python -m src index
```

> **Note**: This requires an OpenAI API call and will cost approximately $0.10-0.50 depending on document size.

## Usage

### Ask a Question

```bash
python -m src ask "What is the Galerkin method?"
```

With filters:
```bash
python -m src ask "Explain CFL condition" --doc-type lecture
python -m src ask "Stability analysis" --doc-type exam --year 2024
```

### Search for Content

Search without generating an AI answer (useful for finding similar problems):

```bash
python -m src search "finite element error estimates"
python -m src search "weak formulation" --doc-type exam
```

Show full content:
```bash
python -m src search "Galerkin" --full
```

### Interactive Chat

Start a multi-turn conversation:

```bash
python -m src chat
```

Commands in chat mode:
- Type your question and press Enter
- `/doc-type lecture` - Filter by document type
- `/year 2024` - Filter by exam year
- `clear` - Clear conversation history
- `quit` or `exit` - End session

### View Statistics

```bash
python -m src stats
```

### Re-index Documents

If you add new documents:

```bash
python -m src index --force
```

## Project Structure

```
NPDE_TA_RAG/
├── data/                    # Course materials (not in git)
│   ├── lectures/
│   ├── homework/
│   └── exams/
├── src/
│   ├── __init__.py
│   ├── cli.py               # Command-line interface
│   ├── config.py            # Configuration
│   ├── pdf_parser.py        # PDF text extraction
│   ├── chunker.py           # Text chunking
│   ├── embeddings.py        # OpenAI embeddings
│   ├── vectorstore.py       # ChromaDB operations
│   ├── retriever.py         # Document retrieval
│   └── llm.py               # GPT-4o integration
├── vectordb/                # Vector database (not in git)
├── requirements.txt
├── .env                     # API keys (not in git)
└── README.md
```

## How It Works

1. **PDF Parsing**: Extracts text from course PDFs while preserving structure
2. **Chunking**: Splits documents into ~1000 token chunks with semantic boundaries
3. **Embedding**: Converts text chunks to vectors using OpenAI's embedding model
4. **Storage**: Stores vectors in ChromaDB for efficient similarity search
5. **Retrieval**: Finds the most relevant chunks for a given query
6. **Generation**: GPT-4o generates answers using retrieved context

## Cost Estimation

| Operation | Approximate Cost |
|-----------|-----------------|
| Initial indexing (~100MB PDFs) | $0.10 - $0.50 |
| Each question (ask/chat) | $0.01 - $0.05 |
| Each search | ~$0.0001 |

## Troubleshooting

### "OPENAI_API_KEY is not set"
Make sure you created the `.env` file with your API key.

### "No documents indexed"
Run `python -m src index` first to index your documents.

### Rate limit errors
You've exceeded your OpenAI API quota. Check your billing at https://platform.openai.com/

## License

This project is for educational use as part of ETH Zurich's NPDE course.

## Acknowledgments

- Course materials by Prof. Ralf Hiptmair, SAM, ETH Zurich
- Built with [ChromaDB](https://www.trychroma.com/), [OpenAI](https://openai.com/), and [Typer](https://typer.tiangolo.com/)
