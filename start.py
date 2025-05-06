import os
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import login
from datasets import load_dataset, concatenate_datasets, load_from_disk
from trl import SFTTrainer, GKDTrainer, GKDConfig
import copy



#log in to huggingface hub
huggingface_token = os.environ.get('HUGGINGFACE_HUB_TOKEN')
if huggingface_token is None:
    raise ValueError("Please set the HUGGINGFACE_HUB_TOKEN environment variable.")
login(token=huggingface_token)


# instantiate teacher and student models
teacher_name = "google/gemma-2-9b-it"
teacher = AutoModelForCausalLM.from_pretrained(
        teacher_name, torch_dtype=torch.float16, device_map="auto",
        attn_implementation="flash_attention_2" #according to HF this needs to be done to avoid NaN in logits
)
tokenizer = AutoTokenizer.from_pretrained(teacher_name)


student_name = "google/gemma-2-2b-it"


student = AutoModelForCausalLM.from_pretrained(
        student_name, torch_dtype=torch.float16, device_map="auto",
        attn_implementation="flash_attention_2" #according to HF this needs to be done to avoid NaN in logits
    )

for p in teacher.parameters():
    p.requires_grad = False
teacher = teacher.eval().to('cuda')


#Starting with initialized embedding matrix and output dense layer 
#not worrying about hidden states for now
# student.get_input_embeddings().load_state_dict(
#         teacher.get_input_embeddings().state_dict()
#     )

# student.lm_head.load_state_dict(teacher.lm_head.state_dict())

# print(student)


#concatenate datasets

xsum_train = load_from_disk("mistral-KD/data/xsum_formatted")
cnn_train = load_from_disk("mistral-KD/data/cnn_formatted")
sci1_train = load_from_disk("mistral-KD/data/sci1_formatted")

train_ds = concatenate_datasets([xsum_train, cnn_train, sci1_train])


#GKD Config
#Starting with no lr scheduler 

train_args = GKDConfig(
    output_dir="mistral-KD/runs/distill",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    num_train_epochs=2,
    learning_rate=5e-5,
    logging_steps=50,
    save_steps=2000,
    fp16=True,
    deepspeed="mistral-KD/deepspeedconfig.json",
    report_to="none",
    warmup_ratio = 0.1,
    lmbda = 0.5, #default 0.5
    beta = 0.5, #default 0.5
    seq_kd=False, #Want to use ground truth labels for now
    dataloader_num_workers=8

)



trainer = GKDTrainer( #default collator set up is good for now
    model=student,
    teacher_model=teacher,
    train_dataset=train_ds,
    args = train_args,
    processing_class=tokenizer
)

trainer.train()