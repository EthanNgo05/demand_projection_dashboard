# Local LLM alternative instead of Claude API

import requests
import json

url = "http://james-workstation:4000/v1/chat/completions"

payload = json.dumps({
  "model": "gemma4-31b",
  "messages": [
    {
      "role": "user",
      "content": "Say hello from the 31B model."
    }
  ]
})
headers = {
  'Content-Type': 'application/json'
}

response = requests.request("POST", url, headers=headers, data=payload)

print(response.text)