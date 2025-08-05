from pydantic import BaseModel

class AnswerItem(BaseModel):
    answer: str
    source: str
    explanation: str

class QueryResponse(BaseModel):
    answers: list[AnswerItem]
