import requests
import json
import os
from dotenv import load_dotenv

load_dotenv()

# First API call with reasoning
response = requests.post(
  url="https://openrouter.ai/api/v1/chat/completions",
  headers={
    "Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}",
    "Content-Type": "application/json",
  },
  data=json.dumps({
    "model": "deepseek/deepseek-v4.1-flash-1017",
    "messages": [
        {
          "role": "user",
          "content": "How many r's are in the word 'strawberry'?"
        }
      ],
    "reasoning": {"enabled": True},
    "provider": {
        "only": [
          "inference-net/us"
        ],
        "allow_fallbacks": False
      }
  })
)

# Extract the assistant message with reasoning_details
resp1 = response.json()
msg = resp1['choices'][0]['message']

# Preserve the assistant message with reasoning_details
messages = [
  {"role": "user", "content": "How many r's are in the word 'strawberry'?"},
  {
    "role": "assistant",
    "content": msg.get('content'),
    "reasoning_details": msg.get('reasoning_details')  # Pass back unmodified
  },
  {"role": "user", "content": "Are you sure? Think carefully."}
]

# Second API call - model continues reasoning from where it left off
response2 = requests.post(
  url="https://openrouter.ai/api/v1/chat/completions",
  headers={
    "Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}",
    "Content-Type": "application/json",
  },
  data=json.dumps({
    "model": "deepseek/deepseek-v4.1-flash-1017",
    "messages": messages,  # Includes preserved reasoning_details
    "reasoning": {"enabled": True},
    "provider": {
        "only": [
          "inference-net/us"
        ],
        "allow_fallbacks": False
      }
  })
)

print("Response 1:", resp1['choices'][0]['message'].get('content'))
print("Response 2:", response2.json()['choices'][0]['message'].get('content'))