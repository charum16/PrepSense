from groq import Groq
from dotenv import load_dotenv
import os

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def test_prompt(strategy_name, messages, temperature=0.7):
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=messages,
        temperature=temperature,
        max_tokens=500,
    )
    result = response.choices[0].message.content
    print(f"\n{'='*50}")
    print(f"Strategy: {strategy_name}")
    print(f"{'='*50}")
    print(result)
    return result

task = "Given the topic 'binary search trees', generate one interview question."

# Strategy 1: Zero-shot
test_prompt("1. Zero-shot", [
    {"role": "user", "content": task}
])

# Strategy 2: Role prompting
test_prompt("2. Role prompting", [
    {"role": "system", "content": "You are a senior software engineer at Google conducting technical interviews."},
    {"role": "user", "content": task}
])

# Strategy 3: Few-shot
test_prompt("3. Few-shot", [
    {"role": "system", "content": "You generate interview questions. Here are examples:\nTopic: arrays -> Question: Given an unsorted array, find the two numbers that sum to a target value.\nTopic: graphs -> Question: How would you detect a cycle in a directed graph?"},
    {"role": "user", "content": task}
])

# Strategy 4: Chain-of-thought
test_prompt("4. Chain-of-thought", [
    {"role": "system", "content": "You are a technical interviewer. Think step by step: first identify the core concept, then think of a practical problem that tests it, then formulate a clear question."},
    {"role": "user", "content": task}
])

# Strategy 5: Structured output
test_prompt("5. Structured output", [
    {"role": "system", "content": "You are a technical interviewer. Always respond in this exact JSON format:\n{\"question\": \"...\", \"difficulty\": \"easy/medium/hard\", \"topic\": \"...\"}"},
    {"role": "user", "content": task}
])