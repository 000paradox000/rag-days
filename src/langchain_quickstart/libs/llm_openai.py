from langchain_openai import ChatOpenAI

from .llm_base import LLMBase


class LLMOpenAI(LLMBase):
    def __init__(self):
        super().__init__()
        self._llm = ChatOpenAI()
