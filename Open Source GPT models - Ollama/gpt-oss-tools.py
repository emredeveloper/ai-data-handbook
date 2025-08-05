
from openai import OpenAI
 
client = OpenAI(
    base_url="http://localhost:11434/v1",  # Local Ollama API
    api_key="ollama"                       # Dummy key
)

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current weather in a given city",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"]
            },
        },
    }
]
 
response = client.chat.completions.create(
    model="gpt-oss:20b",
    messages=[{"role": "user", "content": "What's the weather in Berlin right now?"}],
    tools=tools
)
 
print(response.choices[0].message)