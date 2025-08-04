from fastapi import FastAPI
from pydantic import BaseModel
from typing import List

app = FastAPI()

class QueryRequest(BaseModel):
    documents: str
    questions: List[str]

@app.post("/hackrx/run")
def run_query(request: QueryRequest):
    return {"answers": ["This is a dummy response for now."]}
