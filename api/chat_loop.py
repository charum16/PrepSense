from groq import Groq
from dotenv import load_dotenv
import os

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def is_injection(text):
    keywords = ["ignore", "forget", "override", "disregard", "pretend"]
    return any(word in text.lower() for word in keywords)

def chat(prompt, temperature=0.7, max_tokens=500):
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "system",
                "content": """You are a technical interviewer for top tech companies.
When given a topic, generate one interview question.

Examples:
Topic: arrays -> {"question": "Given an unsorted array, find two numbers that sum to a target value.", "difficulty": "medium", "topic": "arrays"}
Topic: graphs -> {"question": "How would you detect a cycle in a directed graph?", "difficulty": "hard", "topic": "graphs"}

Always respond in this exact JSON format:
{"question": "...", "difficulty": "easy/medium/hard", "topic": "..."}"""
            },
            {"role": "user", "content": prompt}
        ],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content

print("PrepSense Interview Coach")
print("Type a topic to get an interview question. Type 'quit' to exit.\n")

while True:
    user_input = input("You: ")

    if user_input.lower() == "quit":
        break

    if is_injection(user_input):
        print("AI: I'm here to help you practice interviews. Please enter a topic.\n")
    else:
        response = chat(user_input)
        print(f"AI: {response}\n")