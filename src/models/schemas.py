from pydantic import BaseModel, Field
from typing import List, Optional

class ChatRequest(BaseModel):
    query: str = Field(..., description="The user's question or query")

class FAQReference(BaseModel):
    question: str = Field(..., description="The original FAQ question")
    answer: str = Field(..., description="The original FAQ answer")
    similarity_score: float = Field(..., description="Similarity score between query and FAQ")

class ChatResponse(BaseModel):
    answer: str = Field(..., description="The generated answer to the user's query")
    confidence_score: float = Field(..., description="Confidence score of the generated answer")
    faq_references: List[FAQReference] = Field(..., description="List of relevant FAQs used to generate the answer")
    error: Optional[str] = Field(None, description="Error message if something went wrong") 