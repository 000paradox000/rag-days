from time import perf_counter

from dotenv import load_dotenv

load_dotenv(override=False)

from libs.llm_ollama import LLMOllama
from libs.llm_openai import LLMOpenAI


def main():
    question = "Is the creator of Dragon Ball alive?"

    llms = [
        {
            "llm_object": LLMOpenAI(),
            "label": "OpenAI",
        },
        {
            "llm_object": LLMOllama("llama3"),
            "label": "Ollama llama3",
        },
    ]

    print("")
    for llm_item in llms:
        start_time = perf_counter()

        print("=" * 50)
        print(llm_item["label"])
        print("")
        llm_object = llm_item["llm_object"]
        answer = llm_object.ask(question)
        print("-" * 50)
        print(f"Question:\n{question}")
        print("")
        print(f"Answer:\n{answer}")

        end_time = perf_counter()
        execution_time = end_time - start_time
        print("-" * 50)
        print(f"Execution time: {execution_time:.3f}s")
        print("")


if __name__ == "__main__":
    main()
