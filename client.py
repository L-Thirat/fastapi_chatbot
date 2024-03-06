from urllib import parse
import requests
import time


def resp(msg):
    start_time = time.time()
    extra_request = parse.quote(msg)
    url = f'https://4f0ad86a83121.notebooksc.jarvislabs.net/waifuapi?command=chat&data=' + extra_request

    try:
        r = requests.get(url)
        print(r)
        print(r.json())
    except requests.exceptions.ConnectionError as e:
        print(e)
    print("--- %s seconds ---" % (time.time() - start_time))


while True:
    txt = input(">")
    resp(txt)
