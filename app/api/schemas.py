from pydantic import BaseModel, Field


class QueryInput(BaseModel):
    question: str = Field(min_length=3, max_length=2000)
    rewrite: bool = True

class Source(BaseModel):
    subject: str
    sheet: str
    section: str
    source: str
    score: float

class QueryOutput(BaseModel):
    answer: str
    sources: list[Source]
    rewritten_query: str

class Topic(BaseModel):
    title: str
    description: str