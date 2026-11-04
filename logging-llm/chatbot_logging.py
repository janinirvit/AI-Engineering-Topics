import json
import logging
import uuid
from datetime import datetime

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()


def setup_logging():
    """Setup logging configuration."""
    log_filename = "chatbot_logs.json"
    logger = logging.getLogger("chatbot")
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter("%(message)s")
    file_handler = logging.FileHandler(log_filename)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Create a console handler for human -readable logs
    console_handler = logging.StreamHandler()
    console_formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    return logger


def initialize_client() -> OpenAI:
    """Initialize and return the OpenAI client."""
    return OpenAI()


class ChatBot:
    def __init__(self):
        self.client = initialize_client()
        self.logger = setup_logging()
        self.model_name = "gpt-4o-mini"
        self.session_id = str(uuid.uuid4())

        self.messages = [{"role": "system", "content": "You are a helpful customer support assistant."}]

    def chat(self, user_input: str) -> str:
        try:
            log_entry = {
                "timestamp": datetime.now().isoformat(),
                "session_id": self.session_id,
                "level": "INFO",
                "type": "user_input",
                "user_input": user_input,
                "meta_data": {"session_id": self.session_id, "model_name": self.model_name},
            }
            self.logger.info(json.dumps(log_entry))

            self.messages.append({"role": "user", "content": user_input})

            # Here you would typically call the OpenAI API to get the response
            start_time = datetime.now().isoformat()
            response = self.client.chat.completions.create(model=self.model_name, messages=self.messages)
            end_time = datetime.now().isoformat()
            response_time = (datetime.fromisoformat(end_time) - datetime.fromisoformat(start_time)).total_seconds()

            assistance_response = response.choices[0].message.content
            log_entry = {
                "timestamp": datetime.now().isoformat(),
                "level": "INFO",
                "type": "model_response",
                "model_response": assistance_response,
                "meta_data": {
                    "session_id": self.session_id,
                    "model_name": self.model_name,
                    "response_time_seconds": response_time,
                    "token_used": response.usage.total_tokens if hasattr(response.usage, "total_tokens") else None,
                },
            }
            self.logger.info(json.dumps(log_entry))

            self.messages.append({"role": "assistant", "content": assistance_response})
            return assistance_response
        except Exception as e:  # noqa: BLE001
            log_entry = {
                "timestamp": datetime.now().isoformat(),
                "level": "ERROR",
                "type": "error",
                "error_message": str(e),
                "meta_data": {"session_id": self.session_id, "model_name": self.model_name},
            }
        self.logger.error(json.dumps(log_entry))
        return "An error occurred. Please try again."


def main():
    chatbot = ChatBot()

    print("Chatbot initialized. Type 'exit' or 'quit' to end the conversation.")
    print(f"session_id: {chatbot.session_id}")

    while True:
        user_input = input("You: ").strip()

        if user_input.lower() in ["exit", "quit"]:
            print("Exiting the conversation.")
            break
        if not user_input:
            continue

        response = chatbot.chat(user_input)
        print(f"Bot: {response}")


if __name__ == "__main__":
    main()
