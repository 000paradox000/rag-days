class LLMBase:
    def __init__(self):
        self._llm = None

    def ask(self, query: str) -> str:
        return self._llm.invoke(query)
