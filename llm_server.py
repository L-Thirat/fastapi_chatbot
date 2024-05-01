import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, GenerationConfig
import time
from fastapi import FastAPI
from fastapi.responses import JSONResponse

base_model_id = "tanamettpk/TC-instruct-DPO"

# use fast api instead
app = FastAPI()

model = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    torch_dtype=torch.bfloat16,
    # low_gpumem_usage=True,
    return_dict=True,
    device_map={"": 0},
)
tokenizer = AutoTokenizer.from_pretrained(base_model_id)

generation_config = GenerationConfig(
    do_sample=True,
    max_new_tokens=80, temperature=1, top_k=10, top_p=0.95,
    repetition_penalty=1.1,
    pad_token_id=tokenizer.eos_token_id)


@app.get("/llm_api")
async def get_waifuapi(data: str):
    # Tokenize input
    inputs = tokenizer(data, return_tensors="pt").to("cuda")

    # Generate outputs
    st_time = time.time()
    outputs = model.generate(**inputs, generation_config=generation_config)

    # Decode and print response
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"Response time: {time.time() - st_time} seconds")
    return JSONResponse(content={"answer": response})

if __name__ == "_main_":
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