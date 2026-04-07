import requests
import base64
import json
from pathlib import Path

def encode_image(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def ask_llava(image_path, prompt):
    image_b64 = encode_image(image_path)
    
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": "llava",
            "prompt": prompt,
            "images": [image_b64],
            "stream": False
        }
    )
    
    result = response.json()
    return result.get("response", "No response")

if __name__ == "__main__":
    image_path = Path("data/diagrams/test_diagram.png")
    
    if not image_path.exists():
        print("ERROR: test_diagram.png not found in data/diagrams/")
        exit()

    print("Sending image to LLaVA...\n")
    
    prompt = "List every distinct component visible in this diagram. For each component, mention its name and type (e.g. service, database, load balancer, user, etc.)."
    
    response = ask_llava(image_path, prompt)
    
    print("=== LLaVA Response ===")
    print(response)
    