
import json

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

def create_initial_messages() -> list[dict[str, str]]:
    """Create the initial messages for the chat history.

    Returns:
        list[dict[str, str]]: The initial chat messages.
    """
    return [{"role": "system", "content": "You are a helpful assistant."}]

def chat_with_memory(user_input: str, messages: list[dict[str, str]]) -> str:
    """Chat with memory by keeping track of the conversation history.

    Args:
        user_input (str): The user's input message.
        messages (list[dict[str, str]]): The list of previous chat messages.

    Returns:
        str: The model's response.
    """
    client = OpenAI()  # Replace with actual chat client initialization
    model = "gpt-4o-mini"
    try:
        
        messages.append({"role": "user", "content": user_input})

        response = client.chat.completions.create(
            model=model,
            messages=messages
        )
        assistent_response = response.choices[0].message.content
        messages.append({"role": "assistant", "content": assistent_response})
        return assistent_response
    except Exception as e:  # noqa: BLE001
        return f"An error occurred: {e}"
    
def summarize_chat_history(messages: list[dict[str, str]]) -> list[dict[str, str]]:
    """Summarize the chat history into a concise format.

    Args:
        messages (list[dict[str, str]]): The list of chat messages.

    Returns:
        list[dict[str, str]]: The summarized chat messages.
    """
    summary = "Previous conversation summarized: "+ " ".join(
        [m["content"][:50] + "..." for m in messages[-5:]]
    )
    print("Summarized chat history:", summary)
    return [{"role": "system", "content": summary}] + messages[-5:]

def save_conversation(messages: list[dict[str, str]], file_path: str = "conversation.json") -> None:
    """Save the conversation to a file.

    Args:
        messages (list[dict[str, str]]): The list of chat messages.
        file_path (str): The path to the file where the conversation will be saved.
    """
    try:
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(messages, f, ensure_ascii=False, indent=4)
    except Exception as e:  # noqa: BLE001
        print(f"An error occurred while saving the conversation: {e}")
        
def load_conversation(file_path: str = "conversation.json") -> list[dict[str, str]]:
    """Load the conversation from a file.

    Args:
        file_path (str): The path to the file where the conversation is saved.

    Returns:
        list[dict[str, str]]: The list of chat messages.
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:  # noqa: BLE001
        print(f"An error occurred while loading the conversation: {e}")
        return create_initial_messages()
    
def main():
    
    messages = load_conversation()
    while True:
        user_input = input("You: ")
        
        if user_input.lower() in {"exit", "quit"}:
            save_conversation(messages)
            break
        elif user_input.lower() in {"save"}:
            save_conversation(messages)
            print("Conversation saved.")
            continue
        elif user_input.lower() in {"load"}:
            messages = load_conversation()
            print("Conversation loaded.")
            continue
        elif user_input.lower() in {"summary"}:
            messages = summarize_chat_history(messages)
            print("Chat history summarized.")
            continue

        response = chat_with_memory(user_input, messages)
        print(f"Assistant: {response}")
        
        if len(messages) > 10:
            messages = summarize_chat_history(messages)
            print("Chat history summarized due to length.")
            
if __name__ == "__main__":
    main()