import os
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import login
from datasets import load_dataset, concatenate_datasets
from trl import SFTTrainer
from transformers import TrainingArguments

huggingface_token = os.environ.get('HUGGINGFACE_HUB_TOKEN')
if huggingface_token is None:
    raise ValueError("Please set the HUGGINGFACE_HUB_TOKEN environment variable.")
os.environ["HUGGINGFACE_HUB_TOKEN"] = huggingface_token
login(token=huggingface_token)


teacher_name = "mistralai/Mistral-7b-v0.2"
teacher = AutoModelForCausalLM.from_pretrained(
        teacher_name, torch_dtype=torch.float16, device_map="auto",
        token = huggingface_token
    )

tokenizer = AutoTokenizer.from_pretrained(teacher_name)

student_cfg = teacher.config.copy()
student_cfg.num_hidden_layers = 12 

student = AutoModelForCausalLM.from_config(student_cfg)

student.get_input_embeddings().load_state_dict(
        teacher.get_input_embeddings().state_dict()
    )

student.lm_head.load_state_dict(teacher.lm_head.state_dict())


