# =============================================================================
# Common

install-requirements:
	pip install -r requirements.txt

lint:
	pre-commit run --all-files
	pre-commit run --all-files

lint-update:
	pre-commit autoupdate

console:
	ipython

# =============================================================================
# Ollama

run-ollama_chat-with-llama3:
	python src/ollama_chat/main.py

run-ollama_chat-with-gemma:
	python src/ollama_chat/main.py -m gemma:2b

run-ollama_chat-with-phi3mini:
	python src/ollama_chat/main.py -m phi3

run-ollama_chat-with-mistral:
	python src/ollama_chat/main.py -m mistral

run-ollama_chat-with-neuralchat:
	python src/ollama_chat/main.py -m "neural-chat"

# =============================================================================
# RAG with OpenAI

run-rag-with-openai:
	python src/rag_openai/main.py

# =============================================================================
# LangChain Quickstarts

run-langchain-quickstart:
	python src/langchain_quickstart/main.py
