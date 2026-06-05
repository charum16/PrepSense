from groq import Groq
from dotenv import load_dotenv
import os

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def chat(prompt, temperature=0.7, max_tokens=500):
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
    {
        "role": "system",
        "content": "You are a technical interviewer for top tech companies. When the user gives you a topic, ask them one clear interview question on that topic. Be concise and professional."
    },
    {"role": "user", "content": prompt}],

        temperature=temperature,
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content

print("PrepSense Interview Coach - Chat Test")
print("Type 'quit' to exit\n")

while True:
    user_input = input("You: ")
    
    if user_input.lower() == "quit":
        break

    response = chat(user_input)
    print(f"AI: {response}\n")

