from pydantic import BaseModel
from typing import List, Optional

class AnswerItem(BaseModel):
    answer: str
    source: str
    explanation: str
    confidence: Optional[float] = None
    query_processed: Optional[str] = None
    context_chunks_count: Optional[int] = None
    model_used: Optional[str] = None

class QueryResponse(BaseModel):
    answers: List[AnswerItem]
