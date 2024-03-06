from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
import torch
import logging
from fastapi import FastAPI
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)

tokenizer = AutoTokenizer.from_pretrained("PygmalionAI/pygmalion-1.3b", use_fast=True)
config = AutoConfig.from_pretrained("PygmalionAI/pygmalion-1.3b", is_decoder=True)
model = AutoModelForCausalLM.from_pretrained("PygmalionAI/pygmalion-1.3b", config=config, )
#sambanovasystems/SambaLingo-Thai-Chat
use_gpu = torch.cuda.is_available()
logger.debug("Detecting GPU...")
if use_gpu:
    logger.debug("GPU detected!")
    device = torch.device('cuda')
else:
    logger.debug("Using CPU...")
    use_gpu = False
    device = torch.device('cpu')

model = model.to(device)


# use fast api instead
app = FastAPI()


def construct_msg_internal_hist(text: str, role="users") -> str:
    return f"\n{role}: {text.strip()}"


@app.get("/waifuapi")
async def get_waifuapi(data: str):
    msg = construct_msg_internal_hist(data)
    inputs = tokenizer(msg, return_tensors='pt')
    if use_gpu:
        inputs = inputs.to(device)
    print("generate output ..\n")
    out = model.generate(**inputs, max_length=len(inputs['input_ids'][0]) + 50,  # todo must < 280
                         pad_token_id=tokenizer.eos_token_id)
    conversation = tokenizer.decode(out[0])
    print("conversation .. \n" + conversation)
    return JSONResponse(content={"answer": conversation})


if __name__ == "__main__":
    import uvicorn
    import socket

    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    port = 6006
    try:
        s.bind(("localhost", port))
        s.close()
    except socket.error as e:
        print(f"Port {port} is already in use")
        exit()
    uvicorn.run(app, host="0.0.0.0", port=port)
