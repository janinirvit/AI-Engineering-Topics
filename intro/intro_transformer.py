from transformers import pipeline


def create_simple_llm():
    model_name = "distilgpt2"

    # create generator pipeline
    generator = pipeline("text-generation", model=model_name, pad_token_id=50256)

    return generator


generator = create_simple_llm()

prompt = "Once upon a time"

generated_text = generator(prompt, max_length=1000, num_return_sequences=1)

print(generated_text[0]["generated_text"])
