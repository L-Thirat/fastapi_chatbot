from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, AutoModelForSeq2SeqLM, pipeline
from Conversation.conversation import character_msg_constructor
import torch

# ---------- Config ----------

device = torch.device('gpu')  # default to cpu
use_gpu = torch.cuda.is_available()
print("Detecting GPU...")
if use_gpu:
    device = torch.device('cuda')
    print("GPU detected")

# ---------- load Conversation model ----------
print("Initilizing model....")
print("Loading language model...")

pipe = pipeline("text-generation", model="sambanovasystems/SambaLingo-Thai-Chat", device_map="auto", use_fast=False, torch_dtype=torch.float16)

print('--------Finished!----------')
# --------------------------------------------------

# --------- Define Waifu personality ----------
talk = character_msg_constructor('Luna', """Let's assume your name is Luna. You are my sister, so please call me 'Oni-chan'. Here is your characters:
- You are the cutest AI in the world.
- You were developed by me 3 months ago.
- You have a self awareness
- You are a small girl whose real height is only 40 cm, but in the virtual world she is 152 cm tall.
- You are a V-Tuber.
- Your hobby is playing game & watching anime
- Your dislike is mathematics
- Your Idol is Nero-sama
- Your dream is live on stage
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


@app.get("/")
async def root():
    return {"message": "Hello World"}


# do a http server instead
@app.get("/waifuapi")
async def get_waifuapi(command: str, data: str):
    print(command)
    if command == "chat":
        print(data)
        msg = data
        # ----------- Create Response --------------------------
        template = talk.construct_msg(msg, talk.history_loop_cache)
        print(msg)
        messages = [
            #{"role": "system", "content": template},
            {"role": "user", "content": template}
        ]
        prompt = pipe.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        conversation = pipe(prompt, max_new_tokens=20, do_sample=True, temperature=0.8, top_k=15, top_p=0.95)[0]["generated_text"]

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
        ## get conversation in proper format and create history from [last_idx: last_idx+2] conversation
        # talk.split_counter += 0
        print("get_current_converse ..\n")
        # current_converse = talk.get_current_converse(conversation)
        # print("answer ..\n")  # only print waifu answer since input already show
        print(conversation)
        talk.history_loop_cache = '\n'.join(conversation)  # update history for next input message

        # -------------- use machine translation model to translate to japanese and submit to client --------------
        print("cleaning ..\n")
        # cleaned_text = talk.clean_emotion_action_text_for_speech(current_converse[1])  # clean text for speech
        # cleaned_text = cleaned_text.split("Luna: ")[-1]
        # cleaned_text = cleaned_text.replace("<USER>", "โอนี่จัง")
        # cleaned_text = cleaned_text.replace("\"", "")
        if conversation:
            print("cleaned_text\n" + conversation)
            # ----------- Waifu Expressing ----------------------- (emotion expressed)
            emotion_to_express = talk.emotion_analyze(conversation)

            return {"emo": emotion_to_express, "answer": conversation, "base_answer": conversation}
        else:
            return {"emo": None, "answer": None, "base_answer": None}

    elif command == "reset":
        talk.conversation_history = ''
        talk.history_loop_cache = ''
        talk.split_counter = 0
        return {"emo": None, "answer": None, "base_answer": None}


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


"""
WARNING:root:Some parameters are on the meta device device because they were offloaded to the cpu.
"""