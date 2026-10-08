from google import genai
import os
from dotenv import load_dotenv

load_dotenv()

client = genai.Client(api_key=os.getenv("API_KEY"))

history = []

while True:
    prompt = input("Enter your prompt: ")

    if prompt == "exit":
        break

    history.append(
        {
            "type": "user_input",
            "content": [{"type": "text", "text": prompt}],
        }
    )

    interaction = client.interactions.create(
        model="gemini-2.5-flash", store=False, input=history
    )

    print(interaction.output_text)

    for step in interaction.steps:
        history.append(step.model_dump())
