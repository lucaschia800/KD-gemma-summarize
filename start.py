import os
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset, concatenate_datasets
from trl import SFTTrainer
from transformers import TrainingArguments


teacher_name = "mistralai/Mistral-7b-v0.2"
teacher = AutoModelForCausalLM.from_pretrained(
        teacher_name, torch_dtype=torch.float16, device_map="auto"
    )

tokenizer = AutoTokenizer.from_pretrained(teacher_name)

student_cfg = teacher.config.copy()
student_cfg.num_hidden_layers = 12 

print(student.cfg)
