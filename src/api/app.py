from flask import Flask, request, jsonify
from src.models.schemas import ChatRequest, ChatResponse
from src.embeddings.embedding_manager import EmbeddingManager
from src.rag.rag_manager import RAGManager
import os

app = Flask(__name__)

# Initialize managers
embedding_manager = EmbeddingManager()
rag_manager = RAGManager()

# Load FAQs and create index
FAQ_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data', 'faqs.json')
INDEX_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data', 'faiss_index.bin')

# Load or create index
if os.path.exists(INDEX_PATH):
    embedding_manager.load_index(INDEX_PATH)
else:
    embedding_manager.load_faqs(FAQ_PATH)
    embedding_manager.create_index()
    embedding_manager.save_index(INDEX_PATH)

@app.route('/chat', methods=['POST'])
def chat():
    try:
        # Parse request
        data = request.get_json()
        chat_request = ChatRequest(**data)
        
        # Search for relevant FAQs
        faq_references = embedding_manager.search(chat_request.query)
        
        # Generate response
        response = rag_manager.generate_response(chat_request.query, faq_references)
        
        return jsonify(ChatResponse(**response).dict())
    
    except Exception as e:
        return jsonify(ChatResponse(
            answer="",
            confidence_score=0.0,
            faq_references=[],
            error=str(e)
        ).dict()), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000) 