from transformers import AutoTokenizer, pipeline


def create_simple_llm():
    model_name = "distilgpt2"

    # create generator pipeline
    generator = pipeline("text-generation", model=model_name, pad_token_id=50256)

    return generator


def generate_text(generator, prompt, max_length=1000):
    result = generator(prompt, max_length=max_length, num_return_sequences=1, do_sample=True, temperature=0.7)
    return result[0]["generated_text"]


def run_llm_demo():
    print("Running basic LLM demo...")
    generator = create_simple_llm()

    print("Basic LLM demo initialized.")
    print("You can now enter your prompt to generate text.")

    prompt = ["The quick brown fox jumps over the lazy dog.", "Once upon a time", "python programming is"]

    for p in prompt:
        print(f"Prompt: {p}")
        generated_text = generate_text(generator, p)
        print(f"Generated Text: {generated_text}\n")
        input("Press Enter to continue to the next prompt...")


def interactive_demo():
    generator = create_simple_llm()
    print("Interactive demo initialized.")
    print("Type your prompt or (quit) to exit.")
    while True:
        prompt = input("Enter your prompt (or 'quit' to exit): ")
        if prompt.lower() == "quit":
            break
        generated_text = generate_text(generator, prompt)
        print(f"Generated Text: {generated_text}\n")


def explain_process():
    """Explain the LLM process with a simple example"""
    print("How it works:")
    print("1. inpput text -> Tokenization -> Numbers.")
    print("2. Numbers -> Neural Network (Model Processing) -> Predictions.")
    print("3. Predictions -> Text (New Token) -> Output.")

    tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
    text = "Hello World!"
    tokens = tokenizer.encode(text)
    decoded = tokenizer.decode(tokens)

    print("Example Tokenization Process:")
    print(f"Text: {text}")
    print(f"Tokens: {tokens}")
    print(f"Decoded: {decoded}")


if __name__ == "__main__":
    print("Choose a demo:")
    print("1. run basic demo")
    print("2. Interactive Demo")
    print("3. Explain the process")

    choice = input("Enter your choice: ")
    if choice == "1":
        run_llm_demo()
    elif choice == "2":
        interactive_demo()
    elif choice == "3":
        explain_process()
    else:
        print("Invalid choice.")
