import json
import numpy as np
import faiss
from transformers import AutoTokenizer, AutoModel
import torch
from typing import List, Dict, Tuple
import os

class EmbeddingManager:
    def __init__(self, model_name: str = "distilbert-base-uncased"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.index = None
        self.faqs = []
        
    def _generate_embedding(self, text: str) -> np.ndarray:
        """Generate embedding for a single text."""
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            # Use mean pooling
            embeddings = outputs.last_hidden_state.mean(dim=1)
            return embeddings.cpu().numpy()
    
    def load_faqs(self, faq_path: str):
        """Load FAQs from JSON file."""
        with open(faq_path, 'r') as f:
            data = json.load(f)
            self.faqs = data['faqs']
    
    def create_index(self):
        """Create FAISS index from FAQs."""
        if not self.faqs:
            raise ValueError("No FAQs loaded. Call load_faqs() first.")
        
        # Generate embeddings for all questions
        embeddings = []
        for faq in self.faqs:
            embedding = self._generate_embedding(faq['question'])
            embeddings.append(embedding)
        
        # Stack embeddings and create FAISS index
        embeddings = np.vstack(embeddings)
        dimension = embeddings.shape[1]
        
        # Create and train the index
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(embeddings.astype('float32'))
    
    def search(self, query: str, k: int = 3) -> List[Dict]:
        """Search for similar FAQs."""
        if self.index is None:
            raise ValueError("Index not created. Call create_index() first.")
        
        # Generate embedding for query
        query_embedding = self._generate_embedding(query)
        
        # Search in FAISS index
        distances, indices = self.index.search(query_embedding.astype('float32'), k)
        
        # Prepare results
        results = []
        for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
            if idx < len(self.faqs):  # Ensure index is valid
                similarity_score = 1 / (1 + distance)  # Convert distance to similarity score
                results.append({
                    'question': self.faqs[idx]['question'],
                    'answer': self.faqs[idx]['answer'],
                    'similarity_score': float(similarity_score)
                })
        
        return results
    
    def save_index(self, path: str):
        """Save FAISS index to disk."""
        if self.index is None:
            raise ValueError("No index to save. Call create_index() first.")
        faiss.write_index(self.index, path)
    
    def load_index(self, path: str):
        """Load FAISS index from disk."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Index file not found at {path}")
        self.index = faiss.read_index(path) 