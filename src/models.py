from enum import Enum

from pydantic import BaseModel, constr


class QueryTopic(str, Enum):
    content = "content"
    practical = "practical"
    personal = "personal"
    other = "other"


class LLMResponse(BaseModel):
    answer: str
    topic: QueryTopic
