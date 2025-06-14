# RAG-based FAQ Chatbot

A smart chatbot that answers user queries about an e-commerce platform using Retrieval-Augmented Generation (RAG) with FAISS vector database and LangChain.

## Features

- Semantic search using FAISS vector database
- RAG workflow powered by LangChain
- Lightweight LLM (google/flan-t5-base) for response generation
- Structured JSON outputs using Pydantic
- RESTful API built with Flask
- Deployed on Render

## Project Structure

```
rag-faq-chatbot/
├── data/
│   └── faqs.json           # FAQ dataset
├── src/
│   ├── embeddings/         # Embedding generation and storage
│   ├── rag/               # RAG workflow implementation
│   ├── api/               # Flask API implementation
│   └── models/            # Pydantic models
├── tests/                 # Unit tests
├── requirements.txt       # Project dependencies
└── README.md             # Project documentation
```

## Setup Instructions

1. Clone the repository:
```bash
git clone https://github.com/yourusername/rag-faq-chatbot.git
cd rag-faq-chatbot
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Run the application locally:
```bash
python src/api/app.py
```

## API Usage

### Chat Endpoint

Send a POST request to `/chat` with your query:

```bash
curl -X POST http://localhost:5000/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "What is your return policy?"}'
```

Response format:
```json
{
  "answer": "Returns are accepted within 30 days with a receipt.",
  "confidence_score": 0.85,
  "faq_reference": "What is the return policy?"
}
```

## Development

- The FAQ dataset is stored in `data/faqs.json`
- Embeddings are generated using distilbert-base-uncased
- FAISS index is used for efficient similarity search
- LangChain orchestrates the RAG workflow
- Pydantic models ensure structured outputs

## License

MIT License
