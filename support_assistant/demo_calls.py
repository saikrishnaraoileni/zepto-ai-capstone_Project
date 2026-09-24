# demo_calls.py
# just a quick way for me to test 2 example questions and print the json
# response, without having to actually start the uvicorn server separately.
# uses fastapi's built in TestClient thing for this.

from fastapi.testclient import TestClient
from main import app
import json

client = TestClient(app)

# question 1 has the word "delivery" in it so it should trigger retrieval
# question 2 is totally unrelated so it should just get the generic answer
questions_to_try = [
    "What is your delivery policy?",
    "What's the weather like today?",
]

for q in questions_to_try:
    response = client.post("/ask", json={"query": q})
    print("QUESTION:", q)
    print(json.dumps(response.json(), indent=2))
    print()
