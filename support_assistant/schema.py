# schema.py
# this defines what shape our api request/response has to be. pydantic
# just automatically checks the types are right and yells at us if not

from pydantic import BaseModel


class AskRequest(BaseModel):
    query: str


class AskResponse(BaseModel):
    answer: str
    sources: list[str] = []   # empty list by default, for general questions
    confidence: float          # should be between 0 and 1
