from urllib import parse
import requests
import time

msg = "สวัสดี"

start_time = time.time()
extra_request = parse.quote(msg)
url = f'https://c0258c707a001.notebooksb.jarvislabs.net/waifuapi?data=' + extra_request

try:
    r = requests.get(url)
    print(r)
    print(r.json())
except requests.exceptions.ConnectionError as e:
    print(e)
print("--- %s seconds ---" % (time.time() - start_time))
