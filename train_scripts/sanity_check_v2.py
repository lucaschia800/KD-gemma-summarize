from transformers import Trainer, TrainingArguments
import torch
import torch.nn as nn
from datasets import load_dataset, concatenate_datasets, load_from_disk
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import login
import os
from transformers import DataCollatorForSeq2Seq  
from accelerate import Accelerator




huggingface_token = os.environ.get('HUGGINGFACE_HUB_TOKEN')
if huggingface_token is None:
    raise ValueError("Please set the HUGGINGFACE_HUB_TOKEN environment variable.")
login(token=huggingface_token)



student_name = "google/gemma-2-2b-it"

tokenizer = AutoTokenizer.from_pretrained(student_name)



student = AutoModelForCausalLM.from_pretrained(
        student_name, torch_dtype=torch.float16,    
        attn_implementation="eager"
    )


train_ds = load_from_disk("mistral-KD/data/chatml_tokenised")


train_args = TrainingArguments(
    output_dir="/gscratch/stf/lbc800/mistral-KD/runs/KD",
    per_device_train_batch_size=1,
    num_train_epochs=2,
    learning_rate=5e-5,
    logging_steps=50,
    save_steps=2000,
    fp16=True,
    report_to="none",
    warmup_ratio = 0.1,
    gradient_checkpointing=True,

)

collator = DataCollatorForSeq2Seq(
    tokenizer = tokenizer,
    padding=True,
    max_length=1300)

trainer = Trainer(
    args = train_args,
    model=student,
    data_collator=collator,
    train_dataset = train_ds,
    # temperature = 1.5 #starting with > than 1 as we want an emphasis on the model to match overall distribution not just peaks
)


trainer.train()