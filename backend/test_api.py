import requests

with open("data/diagrams/test_diagram.png", "rb") as f:
    response = requests.post(
        "http://localhost:5000/analyze",
        files={"image": f},
        data={"query": "What happens if the Auth Service fails?"}
    )

print(response.json())