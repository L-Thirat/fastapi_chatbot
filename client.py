from urllib import parse
import requests

msg = "สวัสดี"

extra_request = parse.quote(msg)
url = f'http://localhost:6006/waifuapi?data=' + extra_request

try:
    r = requests.get(url)
    print(r.json())
except requests.exceptions.ConnectionError as e:
    print(e)
