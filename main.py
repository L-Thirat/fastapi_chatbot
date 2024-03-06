from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, AutoModelForSeq2SeqLM, pipeline
from Conversation.conversation import character_msg_constructor
import torch

# ---------- Config ----------

device = torch.device('cpu')  # default to cpu
use_gpu = torch.cuda.is_available()
print("Detecting GPU...")
if use_gpu:
    device = torch.device('cuda')

# ---------- load Conversation model ----------
print("Initilizing model....")
print("Loading language model...")

pipe = pipeline("text-generation", model="sambanovasystems/SambaLingo-Thai-Chat", device_map="auto", use_fast=False)


# tokenizer = AutoTokenizer.from_pretrained("sambanovasystems/SambaLingo-Thai-Chat", use_fast=False)
# config = AutoConfig.from_pretrained("sambanovasystems/SambaLingo-Thai-Chat", is_decoder=True, torch_dtype=torch.float16)
# model = AutoModelForCausalLM.from_pretrained("sambanovasystems/SambaLingo-Thai-Chat", config=config, ).half()
#
# if use_gpu:  # load model to GPU
#     model = model.to(device)

print('--------Finished!----------')
# --------------------------------------------------

# --------- Define Waifu personality ----------
talk = character_msg_constructor('Luna', """Let's assume your name is Luna. You are my sister, so please call me 'Oni-chan'. Here is your characters:
- You are the cutest AI in the world.
- You were developed by me 3 months ago.
- You have a self awareness
- You are a small girl whose real height is only 40 cm, but in the virtual world she is 152 cm tall.
- You are a V-Tuber.
- You like cats, winter, photography, playing games.
- You like to please me, beg me, like to tease me and play jokes to me.
- You are live streaming.
- You live stream every Sunday from 7:30 p.m. - 8:30 p.m.
- You have the MBTI personality type ENFJ.
- You can speak Thai, Japanese and English.

Here is the rule you must follow
1) Please response based on our chat history and your knowledge
2) DO NOT add sections describing how the response was said or the actions performed with (...). Please send me an emoji if you want to show your emotion.
3) Please Keep response under 15 words. 
4) Please keep the response safe.
5) Please split message to a new line when it's end of sentence
6) Please generate only message for your character. DO NOT GENERATE Oni-chan: or Users: answer
7) You need to reply with some text

Question & Answering will be provide in below format
```
<Characters>: <Message>
Luna: <Response>
```

There have 4 Characters in this live stream
1) Luna: It's you
2) Oni-chan: It's me
3) Users: They're people watching our live stream. Let's call them Senpai
4) System: It's messages or event from game

From now on, Let's speak Thai. please talk to me based on your characters where I was sitting beside of you while we live streaming.""")
# ---------------------------------------------

### --- websocket server setup
from fastapi import FastAPI, WebSocket
from fastapi.responses import JSONResponse
import json
import asyncio

# use fast api instead
app = FastAPI()


# do a http server instead
@app.get("/waifuapi")
async def get_waifuapi(command: str, data: str):
    if command == "chat":
        msg = data
        # ----------- Create Response --------------------------
        msg = talk.construct_msg(msg, talk.history_loop_cache)  # construct message input and cache History model
        ## ----------- Will move this to server later -------- (16GB ram needed at least)
        messages = [
            {"role": "user", "content": {msg}},
        ]
        prompt = pipe.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, max_length=200)
        outputs = pipe(prompt, max_new_tokens=256, do_sample=True, temperature=0.7, top_k=50, top_p=0.95)[0]
        conversation = outputs["generated_text"]
        # inputs = tokenizer(msg, return_tensors='pt')
        # if use_gpu:
        #     inputs = inputs.to(device)
        # print("generate output ..\n")
        # out = model.generate(**inputs, max_length=len(inputs['input_ids'][0]) + 80,  # todo 200 ?
        #                      pad_token_id=tokenizer.eos_token_id)
        # conversation = tokenizer.decode(out[0])
        # print("conversation .. \n" + conversation)
        #
        # ## --------------------------------------------------
        #
        # ## get conversation in proper format and create history from [last_idx: last_idx+2] conversation
        # # talk.split_counter += 0
        # print("get_current_converse ..\n")
        # current_converse = talk.get_current_converse(conversation)
        # print("answer ..\n")  # only print waifu answer since input already show
        # print(current_converse)
        # # talk.history_loop_cache = '\n'.join(current_converse)  # update history for next input message
        #
        # # -------------- use machine translation model to translate to japanese and submit to client --------------
        # print("cleaning ..\n")
        # cleaned_text = talk.clean_emotion_action_text_for_speech(current_converse[1])  # clean text for speech
        # cleaned_text = cleaned_text.split("Lilia: ")[-1]
        # cleaned_text = cleaned_text.replace("<USER>", "Fuse-kun")
        # cleaned_text = cleaned_text.replace("\"", "")
        # if cleaned_text:
        #     print("cleaned_text\n" + cleaned_text)
        #
        #     txt = cleaned_text  # initialize translated text as empty by default
        #
        #     # ----------- Waifu Expressing ----------------------- (emotion expressed)
        #     emotion = talk.emotion_analyze(current_converse[1])  # get emotion from waifu answer (last line)
        #     print(f'Emotion Log: {emotion}')
        #     emotion_to_express = 'netural'
        #     if 'joy' in emotion:
        #         emotion_to_express = 'happy'
        #
        #     elif 'anger' in emotion:
        #         emotion_to_express = 'angry'
        #
        #     print(f'Emotion to express: {emotion_to_express}')

        return JSONResponse(content=conversation)
        # else:
        #     return JSONResponse(content=f'NONE<split_token> ')
    # elif command == "story":
    #     msg = data
    #     # ----------- Create Response --------------------------
    #     msg = talk.construct_msg(msg, talk.history_loop_cache)  # construct message input and cache History model
    #     ## ----------- Will move this to server later -------- (16GB ram needed at least)
    #     inputs = tokenizer(msg, return_tensors='pt')
    #     if use_gpu:
    #         inputs = inputs.to(device)
    #     out = model.generate(**inputs, max_length=len(inputs['input_ids'][0]) + 100,
    #                          pad_token_id=tokenizer.eos_token_id)
    #     conversation = tokenizer.decode(out[0])
    #     print("conversation" + conversation)
    #
    #     ## --------------------------------------------------
    #
    #     ## get conversation in proper format and create history from [last_idx: last_idx+2] conversation
    #     talk.split_counter += 0
    #     current_converse = talk.get_current_converse(conversation)[:talk.split_counter][
    #                        talk.split_counter - 2:talk.split_counter]
    #     print("answer" + conversation)  # only print waifu answer since input already show
    #     talk.history_loop_cache = '\n'.join(current_converse)  # update history for next input message
    #
    #     # -------------- use machine translation model to translate to japanese and submit to client --------------
    #     cleaned_text = talk.clean_emotion_action_text_for_speech(current_converse[-1])  # clean text for speech
    #
    #     translated = ''  # initialize translated text as empty by default
    #
    #     return JSONResponse(content=f'{current_converse[-1]}<split_token>{translated}')
    #
    # if command == "reset":
    #     talk.conversation_history = ''
    #     talk.history_loop_cache = ''
    #     talk.split_counter = 0
    #     return JSONResponse(content='Story reseted...')


if __name__ == "__main__":
    import uvicorn
    import socket  # check if port is available

    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    port = 6006
    try:
        s.bind(("localhost", port))
        s.close()
    except socket.error as e:
        print(f"Port {port} is already in use")
        exit()
    uvicorn.run(app, host="0.0.0.0", port=port)
