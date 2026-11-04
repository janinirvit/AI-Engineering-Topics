from openai import OpenAI


def simple_chat_without_memory(user_input: str) -> str:
    """_summary_

    Args:
        user_input (str): _description_

    Returns:
        str: _description_
    """
    client = OpenAI()  # Replace with actual chat client initialization
    model = "gpt-4o-mini"
    try:
        response = client.chat.completions.create(model=model, messages=[{"role": "user", "content": user_input}])
        return response.choices[0].message.content
    except Exception as e:
        return f"An error occurred: {e}"


def main():
    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in {"exit", "quit"}:
            import sys

            sys.exit()
            break
        if user_input == "clear":
            print("\033c", end="")  # Clear the console screen
            continue
        if not user_input:
            continue

        response = simple_chat_without_memory(user_input)
        print(f"Bot: {response}")

        print("\n")


if __name__ == "__main__":
    main()
