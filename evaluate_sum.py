import evaluate, transformers, torch
from datasets import load_dataset
import os
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import login

huggingface_token = os.environ.get('HUGGINGFACE_HUB_TOKEN')
if huggingface_token is None:
    raise ValueError("Please set the HUGGINGFACE_HUB_TOKEN environment variable.")
login(token=huggingface_token)


teacher_name = "google/gemma-2-9b-it"
teacher = AutoModelForCausalLM.from_pretrained(
        teacher_name, torch_dtype=torch.float16, device_map="auto",
        token = huggingface_token
    )

tokenizer = AutoTokenizer.from_pretrained(teacher_name)

