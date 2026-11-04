from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI()
model = "gpt-4o-mini"

# few shot prompting example
few_shot_completion = client.chat.completions.create(
    model=model,
    messages=[
        {"role": "system", "content": "You are a translator."},
        {
            "role": "user",
            "content": """Translate the following text
            'Hello' -> 'Hola'
            'Goodbye' -> 'Adiós'
            Now translate 'Thank you'

            """,
        },
    ],
)

print(few_shot_completion.choices[0].message.content)

# Direct prompt example with openai
direct_completion = client.chat.completions.create(
    model=model, messages=[{"role": "user", "content": "Translate 'Good morning' to Spanish."}]
)

print(direct_completion.choices[0].message.content)

# chain of thought prompting example
chain_of_thought_completion = client.chat.completions.create(
    model=model,
    messages=[
        {"role": "system", "content": "You are a Math tutor."},
        {
            "role": "user",
            "content": """Solve this math problem step by step:
            If john has 3 apples and gives 1 to mary,
            how many apples does john have left?
            """,
        },
    ],
)

print(chain_of_thought_completion.choices[0].message.content)

# Instruction Prompting example
instruction_completion = client.chat.completions.create(
    model=model,
    messages=[
        {"role": "system", "content": "You are an assistant that follows instructions carefully."},
        {
            "role": "user",
            "content": "Write a 300-word essay about the importance of learning programming. use bullet points.",
        },
    ],
)

print(instruction_completion.choices[0].message.content)

# Role Playing example
role_playing_completion = client.chat.completions.create(
    model=model,
    messages=[
        {"role": "system", "content": "You are a friendly shopkeeper."},
        {"role": "user", "content": "I would like to buy some apples."},
    ],
)

print(role_playing_completion.choices[0].message.content)

# Open Ended prompt Example
open_ended_completion = client.chat.completions.create(
    model=model, messages=[{"role": "user", "content": "Tell me an interesting fact about space."}]
)

print(open_ended_completion.choices[0].message.content)

# tempratur and Top p sampling example
temperature_top_p_completion = client.chat.completions.create(
    model=model,
    messages=[{"role": "user", "content": "Generate a creative story about a dragon."}],
    temperature=0.5,  # controls the randomness of the output
    # top_p=0.8 # controls the diversity of the output
)

print(temperature_top_p_completion.choices[0].message.content)

# Combination techniques
combination_completion = client.chat.completions.create(
    model=model,
    messages=[
        {"role": "system", "content": "You are a travel blogger."},
        {
            "role": "user",
            "content": (
                "Write a 500-word blog post about your recent trip to melbourne, Australia. "
                "Make sure to give step by step itinary for trip."
            ),
        },
    ],
    temperature=0.7,  # controls the randomness of the output
    # top_p=0.9 # controls the diversity of the output
    stream=True,
)

# print(combination_completion.choices[0].message.content)
for chunk in combination_completion:
    if chunk.choices[0].delta.content is not None:
        print(chunk.choices[0].delta.content, end="")
        print("\n")
