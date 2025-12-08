import os

from openai import OpenAI


def test_openrouter_api(prompt: str) -> str:
    """Test function to check if OpenRouter API is working."""
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OpenRouter API key is required")
    
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1", 
        api_key=api_key
    )
    
    response = client.chat.completions.create(
        model="meta-llama/llama-3.1-8b-instruct",
        messages=[{"role": "user", "content": prompt}],
        temperature=1.0,
    )
    
    return response.choices[0].message.content

if __name__ == "__main__":
    prompt = "What is the capital of Sweden?"
    response = test_openrouter_api(prompt)
    print(response)