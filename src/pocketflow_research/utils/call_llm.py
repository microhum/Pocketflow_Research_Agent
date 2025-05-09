import requests
import os
import json
from dotenv import load_dotenv

# utils/call_llm.py

def call_llm(prompt: str) -> str:
    """
    Placeholder function for calling a Large Language Model.

    Args:
        prompt (str): The input prompt for the LLM.

    Returns:
        str: A response from the LLM.
    """
    print(f"--- Sending prompt to LLM --- \n{prompt}\n---------------------------")
    load_dotenv()
    api_key = os.environ.get("OPENROUTER_API_KEY")
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": "meta-llama/llama-3.3-70b-instruct",
        "messages": [{"role": "user", "content": prompt}]
    }
    
    response = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers=headers,
        data=json.dumps(payload)
    )
    
    if response.status_code == 200:
        return response.json()["choices"][0]["message"]["content"]
    else:
        return f"Error: {response.status_code}, {response.text}"

if __name__ == "__main__":
    test_prompt = "What is the capital of France?"
    print(f"Testing call_llm with prompt: '{test_prompt}'")
    response = call_llm(test_prompt)
    print(f"Received response: {response}")
