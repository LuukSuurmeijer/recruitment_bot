"""
    The Chatbot makes use of Structured Generation. This means that the LLM output can be guided to adhere to some desired syntax. 
    If you're interested: This works by applying a finite-state-automaton to the output probabilities, and setting the probability mass of unreachable states to 0.
    In this case, as a default, I opted to use simple JSON. The LLM will respond with a Pydantic object (LLMResponse) as described below
"""

from enum import Enum
from pydantic import BaseModel


class QueryTopic(str, Enum):
    content = "content"
    practical = "practical"
    personal = "personal"
    other = "other"


class LLMResponse(BaseModel):
    answer: str
    topic: QueryTopic
