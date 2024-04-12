from urllib import parse
import requests
import time


def resp(msg):
    start_time = time.time()
    msg = "Oni-chan: " + msg + "</s>"
    extra_request = parse.quote(msg)
    print(extra_request)
    url = f'https://fbda38e825481.notebooksa.jarvislabs.net/waifuapi?command=chat&data=' + extra_request
    # url = f'https://78edd34dcc8a1.notebooksa.jarvislabs.net/waifuapi?command=chat&data=' + extra_request

    try:
        r = requests.get(url)
        print(r)
        print(r.json()["answer"])
    except requests.exceptions.ConnectionError as e:
        print(e)
    print("--- %s seconds ---" % (time.time() - start_time))


while True:
    txt = input(">")
    resp(txt)

# from lib.audio import transcribe_audio_data, press_speak_capture, write_audio_data
# def listening_loop():
#     while True:
#         time.sleep(0.5)
#         if keyboard.is_pressed('LEFT_ALT'):
#             audio_data = press_speak_capture()
#             if audio_data:
#                 transcription = transcribe_audio_data(audio_data, Characters.CHAR2.tgt_lang)
#                 if transcription is not None:
#                     add_stack_con(transcription)
#                     write_audio_data(audio_data)
# selected_thread.append(threading.Thread(target=listening_loop))