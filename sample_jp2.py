from transformers import  AutoModelForCausalLM
import torch


def infer_flash_compile(model, tokenizer):
    model.to_bettertransformer()
    token_ids=tokenizer.encode(
        prompt,

    )