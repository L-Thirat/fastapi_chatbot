from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, AutoModelForSeq2SeqLM, pipeline
from Conversation.conversation import character_msg_constructor
import torch

# ---------- Config ----------
torch.cuda.empty_cache()

device = torch.device('cuda')  # default to cpu
use_gpu = torch.cuda.is_available()
print("Detecting GPU...")
if use_gpu:
    device = torch.device('cuda')
    print("GPU detected")


# ---------- load Conversation model ----------
print("Initilizing model....")
print("Loading language model...")

pipe = pipeline("text-generation", model="cyberagent/calm2-7b-chat", device_map="auto", use_fast=False, torch_dtype=torch.float16)
# pipe = pipeline("text-generation", model="sambanovasystems/SambaLingo-Thai-Chat", device_map="auto", use_fast=False, torch_dtype=torch.float16)
pipe.model.half()

print('--------Finished!----------')
# --------------------------------------------------

# --------- Define Waifu personality ----------
talk = character_msg_constructor('Luna', """Let's roleplay. Your name is Luna. You are my sister, so please call me 'Oni-chan'. Here is your characters:
- You are the cutest AI in the world.
- You are a V-Tuber.
- You are live streaming

Here is the rule you must follow
1) Please response based on our chat history and your knowledge
2) Please speak with me casually
3) Please Keep response under 15 words. 

Other than you can Oni-chan, there also have 2 more characters
1) Users: They're people watching our live stream. Let's call them Senpai
2) System: It's messages or event from game

From now on, Let's speak Japanese.</s>""")
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
        # ----------- Create Response --------------------------
        talk.construct_msg(data)

        prompt = pipe.tokenizer.apply_chat_template(talk.history_loop_cache, tokenize=False, add_generation_prompt=True)
        conversation = pipe(prompt,
                            max_new_tokens=30,
                            do_sample=True,
                            temperature=0.8,
                            top_k=20,
                            top_p=0.8
                            )[0]["generated_text"]
        answer = conversation[len(prompt):].strip()

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
        # current_converse = talk.get_current_converse(data, conversation)
        # print("answer ..\n")  # only print waifu answer since input already show
        print(answer)
        # talk.history_loop_cache += '\n'.join(current_converse)  # update history for next input message
        talk.history_loop_cache.append({"role": "assistant", "content": answer})

        # -------------- use machine translation model to translate to japanese and submit to client --------------
        print("cleaning ..\n")
        # cleaned_text = talk.clean_emotion_action_text_for_speech(current_converse[1])  # clean text for speech
        # cleaned_text = cleaned_text.split("Luna: ")[-1]
        # cleaned_text = cleaned_text.replace("<USER>", "โอนี่จัง")
        # cleaned_text = cleaned_text.replace("\"", "")
        # if current_converse:
        #     print("cleaned_text\n" + current_converse[1])
        #     # ----------- Waifu Expressing ----------------------- (emotion expressed)
        #     emotion_to_express = talk.emotion_analyze(current_converse[1])

        return {"emo": "emotion_to_express", "answer": answer, "base_answer": answer}
        # else:
        #     return {"emo": None, "answer": None, "base_answer": None}

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