import sys, os, json
sys.path.append('.')
from groq import Groq
from dotenv import load_dotenv
load_dotenv()

client = Groq(api_key=os.getenv('GROQ_API_KEY'))

response = client.chat.completions.create(
    model='llama-3.3-70b-versatile',
    messages=[
        {'role': 'system', 'content': 'You are an interviewer. Respond in JSON only. No text before or after.'},
        {'role': 'user', 'content': 'Generate a DSA interview question about arrays. Include exactly 2 input/output examples.\n\nRespond in this exact JSON:\n{"question": "...", "difficulty": "easy", "topic": "...", "examples": [{"input": "...", "output": "..."}, {"input": "...", "output": "..."}]}'}
    ],
    temperature=0.7,
    max_tokens=600,
    stream=False
)

raw = response.choices[0].message.content
print('RAW RESPONSE:')
print(raw)
print()

try:
    cleaned = raw.strip().removeprefix('```json').removeprefix('```').removesuffix('```').strip()
    parsed = json.loads(cleaned)
    print('PARSED EXAMPLES:', parsed.get('examples'))
except Exception as e:
    print('PARSE ERROR:', e)