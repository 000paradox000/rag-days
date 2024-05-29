from langchain_community.llms import Ollama

from .llm_base import LLMBase


class LLMOllama(LLMBase):
    def __init__(self, model_name: str = "llama3"):
        super().__init__()
        self._model_name = model_name
        self._llm = Ollama(model=self._model_name)
