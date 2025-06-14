from langchain.llms import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from transformers import AutoModelForSeq2SeqGeneration, AutoTokenizer
import torch
from typing import List, Dict
from src.models.schemas import FAQReference

class RAGManager:
    def __init__(self, model_name: str = "google/flan-t5-base"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize the model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqGeneration.from_pretrained(model_name).to(self.device)
        
        # Create the pipeline
        self.pipeline = HuggingFacePipeline(
            pipeline={
                "model": self.model,
                "tokenizer": self.tokenizer,
                "max_length": 512,
                "temperature": 0.7,
                "top_p": 0.95,
                "repetition_penalty": 1.15
            }
        )
        
        # Create the prompt template
        self.prompt_template = PromptTemplate(
            input_variables=["context", "question"],
            template="""Based on the following context, answer the question. If the context doesn't contain enough information, say so.

Context:
{context}

Question: {question}

Answer:"""
        )
        
        # Create the chain
        self.chain = LLMChain(llm=self.pipeline, prompt=self.prompt_template)
    
    def _format_context(self, faq_references: List[Dict]) -> str:
        """Format FAQ references into a context string."""
        context_parts = []
        for ref in faq_references:
            context_parts.append(f"Q: {ref['question']}\nA: {ref['answer']}")
        return "\n\n".join(context_parts)
    
    def generate_response(self, query: str, faq_references: List[Dict]) -> Dict:
        """Generate a response using the RAG workflow."""
        # Format the context
        context = self._format_context(faq_references)
        
        # Generate the response
        response = self.chain.run(context=context, question=query)
        
        # Calculate confidence score (average of similarity scores)
        confidence_score = sum(ref['similarity_score'] for ref in faq_references) / len(faq_references)
        
        # Convert FAQ references to Pydantic models
        faq_refs = [
            FAQReference(
                question=ref['question'],
                answer=ref['answer'],
                similarity_score=ref['similarity_score']
            )
            for ref in faq_references
        ]
        
        return {
            "answer": response.strip(),
            "confidence_score": confidence_score,
            "faq_references": faq_refs
        } 