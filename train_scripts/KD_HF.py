from transformers import Trainer, TrainingArguments
import torch
import torch.nn as nn
from datasets import load_dataset, concatenate_datasets, load_from_disk
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import login
import os
from transformers import DataCollatorForSeq2Seq  



huggingface_token = os.environ.get('HUGGINGFACE_HUB_TOKEN')
if huggingface_token is None:
    raise ValueError("Please set the HUGGINGFACE_HUB_TOKEN environment variable.")
login(token=huggingface_token)


class KDTrainer(Trainer):
    def __init__(self, teacher_model, temperature=1.0, alpha=0.5, **kwargs):
        super().__init__(**kwargs)
        _teacher = teacher_model
        _teacher.eval()
        self.teacher = self.accelerator.prepare_model(_teacher, evaluation_mode=True)
        self.temperature = temperature
        self.alpha = alpha  # Weight for the soft loss

    def compute_loss(self, model, inputs, return_outputs=False, **ignored):
        print(f"Student model device: {model.device}")  
        print(f"Teacher model device: {self.teacher.device}")  
        labels = inputs.get("labels").to(model.device)
        input_ids = inputs.get("input_ids").to(model.device)
        attention_mask = inputs.get("attention_mask", None)
        if attention_mask is not None:
            attention_mask = attention_mask.to(model.device)

        with torch.no_grad():
            teacher_outputs = self.teacher(
                input_ids=input_ids, attention_mask=attention_mask
            ).logits

        

        student_outputs = model(input_ids=input_ids, attention_mask=attention_mask).logits

        # Soft loss
        T = self.temperature
        soft_teacher = nn.functional.softmax(teacher_outputs / T, dim=-1)
        soft_student = nn.functional.log_softmax(student_outputs / T, dim=-1)
        kd_loss = nn.KLDivLoss(reduction="batchmean")(soft_student, soft_teacher) * (T**2)  #reverse KL

        # Hard loss
        ce_loss = nn.CrossEntropyLoss()(student_outputs.view(-1, student_outputs.size(-1)), labels.view(-1))

        loss = self.alpha * kd_loss + (1 - self.alpha) * ce_loss

        return (loss, student_outputs) if return_outputs else loss


teacher_name = "google/gemma-2-9b-it"
teacher = AutoModelForCausalLM.from_pretrained(
        teacher_name, torch_dtype=torch.float16,
        attn_implementation="eager" #according to HF this needs to be done to avoid NaN in logits
    )

tokenizer = AutoTokenizer.from_pretrained(teacher_name)


student_name = "google/gemma-2-2b-it"


student = AutoModelForCausalLM.from_pretrained(
        student_name, torch_dtype=torch.float16,    
        attn_implementation="eager"
    )




train_ds = load_from_disk("mistral-KD/data/chatml_tokenised")


train_args = TrainingArguments(
    output_dir="/gscratch/stf/lbc800/mistral-KD/runs/KD",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    num_train_epochs=2,
    learning_rate=5e-5,
    logging_steps=50,
    save_steps=2000,
    fp16=True,
    deepspeed="/gscratch/stf/lbc800/mistral-KD/deepspeedconfig.json",
    report_to="none",
    warmup_ratio = 0.1,
    gradient_checkpointing=True

)

collator = DataCollatorForSeq2Seq(
    tokenizer,
    padding=True,
    max_length=1300,
    truncatiuon=True,
)

trainer = KDTrainer(
    args = train_args,
    teacher_model=teacher,
    model=student,
    data_collator=collator,
    processing_class=tokenizer,
    train_dataset = train_ds,
    temperature = 1.5 #starting with > than 1 as we want an emphasis on the model to match overall distribution not just peaks
)

trainer.train()