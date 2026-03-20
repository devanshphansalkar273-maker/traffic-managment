import requests

response = requests.post(
    "http://127.0.0.1:5000/set_decision",
    json={
        "action": "GIVE_GREEN",
        "lane": "north"
    }
)

print(f"Status Code: {response.status_code}")
print(f"Response: {response.json()}")
