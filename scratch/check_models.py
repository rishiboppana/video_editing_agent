import requests
try:
    resp = requests.get("http://localhost:11434/api/tags")
    if resp.status_code == 200:
        models = [m['name'] for m in resp.json().get('models', [])]
        print(f"Models: {models}")
    else:
        print(f"Error: {resp.status_code}")
except Exception as e:
    print(f"Fail: {e}")
