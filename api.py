from typing import List

from fastapi import FastAPI

from query_builder import query_suggestion

from pydantic import BaseModel

app = FastAPI()

# Define a Pydantic model for the input
class PhraseRequest(BaseModel):
    phrase: str

# Endpoint to process a phrase and return a list of query suggestions
@app.post("/process", response_model=List[str])
async def process_string_endpoint(request: PhraseRequest):
    """
    Endpoint to process a phrase (from JSON input) and return a list of query suggestions.
    """
    phrase = request.phrase
    return query_suggestion(phrase)
