import argparse
from time import perf_counter

import ollama


def parse_cli_arguments():
    print("")

    parser = argparse.ArgumentParser()

    # Add an argument for the model
    parser.add_argument(
        "-m",
        "--model_name",
        default="llama3",
        help="Name of the model",
    )

    # Add an argument for the question
    parser.add_argument(
        "-q",
        "--question",
        default="Is Dragon Ball creator alive?",
        help="Question to ask",
    )

    # Parse the command-line arguments
    args = parser.parse_args()

    return args.model_name, args.question


def main():
    start_time = perf_counter()

    model_name, question = parse_cli_arguments()

    # check if model is already pulled
    try:
        ollama.chat(model_name)
    except ollama.ResponseError as e:
        if e.status_code == 404:
            print(f"The model '{model_name}' was not found, trying to pull it")
            ollama.pull(model_name)
            print(f"The model '{model_name}' was pulled successfully")

    response = ollama.chat(
        model=model_name,
        messages=[
            {
                "role": "user",
                "content": question,
            },
        ],
    )

    answer = response["message"]["content"]

    print("-" * 50)

    print("Question:")
    print(question)
    print("")
    print("Answer:")
    print(answer)

    end_time = perf_counter()
    execution_time = end_time - start_time

    print("-" * 50)

    print(f"Execution time: {execution_time:.3f}s")

    print("")


if __name__ == "__main__":
    main()
