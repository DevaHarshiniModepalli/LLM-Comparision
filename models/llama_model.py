from huggingface_hub import InferenceClient
import os

def llama_response(prompt: str) -> str:
    token = os.getenv("llama_key")
    if not token:
        return " HF_API_KEY not found"

    client = InferenceClient(
        model="meta-llama/Meta-Llama-3-8B-Instruct",
        token=token
    )

    response = client.chat_completion(
        messages=[
            {"role": "user", "content": prompt}
        ],
        max_tokens=300,
        temperature=0.7
    )

    return response.choices[0].message["content"]