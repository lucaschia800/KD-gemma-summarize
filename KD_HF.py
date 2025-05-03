from transformers import Trainer, TrainingArguments
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from datasets import load_dataset, concatenate_datasets, load_from_disk
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import login
import os


huggingface_token = os.environ.get('HUGGINGFACE_HUB_TOKEN')
if huggingface_token is None:
    raise ValueError("Please set the HUGGINGFACE_HUB_TOKEN environment variable.")
login(token=huggingface_token)


class KDTrainer(Trainer):
    def __init__(self, teacher_model, temperature=1.0, alpha=0.5, **kwargs):
        super().__init__(**kwargs)
        self.teacher = teacher_model
        self.teacher.eval()
        self.temperature = temperature
        self.alpha = alpha  # Weight for the soft loss

    def compute_loss(self, model, inputs, return_outputs=False):
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
        kd_loss = nn.KLDivLoss(reduction="batchmean")(soft_student, soft_teacher) * (T**2)

        # Hard loss
        ce_loss = nn.CrossEntropyLoss()(student_outputs.view(-1, student_outputs.size(-1)), labels.view(-1))

        loss = self.alpha * kd_loss + (1 - self.alpha) * ce_loss

        return (loss, student_outputs) if return_outputs else loss


teacher_name = "google/gemma-2-9b-it"
teacher = AutoModelForCausalLM.from_pretrained(
        teacher_name, torch_dtype=torch.float16, device_map="auto",
        token = huggingface_token
    )

tokenizer = AutoTokenizer.from_pretrained(teacher_name)


student_name = "google/gemma-2-2b-it"


student = AutoModelForCausalLM.from_pretrained(
        student_name, torch_dtype=torch.float16, device_map="auto",
        token = huggingface_token
    )


xsum_train = load_from_disk("mistral-KD/data/xsum_formatted")
cnn_train = load_from_disk("mistral-KD/data/cnn_formatted")
sci1_train = load_from_disk("mistral-KD/data/sci1_formatted")

train_ds = concatenate_datasets([xsum_train, cnn_train, sci1_train])


train_args = TrainingArguments(
    output_dir="mistral_KD/runs/distill",
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

)