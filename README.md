# RAG Days

This is just some PoCs regarding:

- LLMs
- LangChain
- Ollama
- RAG

## Ollama Chat

This is a simple script that allow you ask a question to any model supported
by Ollama.

Source code: src/ollama_chat

```shell
python src/ollama_chat/main.py [-m MODEL_NAME] [-q QUESTION]

python src/ollama_chat/main.py -m gemma:2b

python src/ollama_chat/main.py -m gemma:2b -q "who is the creator of dragon ball?"
```

There are some rules in the Makefile:

```shell
make run-ollama_chat-with-gemma

make run-ollama_chat-with-llama3
```

## RAG with OpenAI, FAISS, LangChain

This is a simple script that allow you conversate with a PDF

Source code: src/rag_openai

```shell
make run-rag-with-openai
```

## LangChain Quickstart

https://python.langchain.com/v0.1/docs/get_started/quickstart/

Source code: src/langchain_quickstart

```shell
make run-langchain-quickstart
```
