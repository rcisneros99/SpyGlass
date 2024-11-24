import requests

url = "https://api.play.ht/api/v2/voices"

headers = {"accept": "application/json"}

response = requests.get(url, headers=headers)

print(response.text)