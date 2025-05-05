import transformers, torch
from datasets import load_dataset, load_from_disk
import os
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import login
from evaluate import evaluator
import json

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
test_ds = load_from_disk("mistral-KD/data/cnn_eval")


task_evaluator = evaluator("summarization")

results = task_evaluator.compute(
    model = teacher,
    metric = "rouge",
    tokenizer = tokenizer,
    data = test_ds,
    strategy = "bootstrap",
    n_resamples = 250,
    label_colum = "summary",
    input_column = "article"
)


print(results)

with open("eval_results/teacher_rouge_results.json", "w") as f:
    json.dump(results, f, indent=4)


