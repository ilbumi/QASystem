from typing import Dict, List, Optional, Union

from pydantic import BaseConfig, BaseModel, Extra, Field


class RequestBaseModel(BaseModel):
    class Config:
        # Forbid any extra fields in the request to avoid silent failures
        extra = Extra.forbid


class QueryRequest(RequestBaseModel):
    question: str
    params: Optional[dict] = None


class Answer(BaseModel):
    answer: str
    context: str
    score: float


class QueryResponse(BaseModel):
    question: str
    results: List[Answer]
