import os
import transformers, torch
from datasets import load_dataset, load_from_disk
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, pipeline
from huggingface_hub import login
import evaluate
import json
import torch._dynamo as dynamo
from tqdm import tqdm

torch.set_float32_matmul_precision("high")
dynamo.config.cache_size_limit = 64

huggingface_token = os.environ.get('HUGGINGFACE_HUB_TOKEN')
if huggingface_token is None:
    raise ValueError("Please set the HUGGINGFACE_HUB_TOKEN environment variable.")

login(token=huggingface_token)




def evaluate_rouge(model, tokenizer, dataset):

    task_evaluator = evaluate.load("rouge")

    def collator(batch):
        # Tokenize the input text and labels

        articles = [example["article"] for example in batch]
        summaries = [example["summary"] for example in batch]

        inputs = tokenizer(
            articles,
            max_length=1500,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        inputs['summary'] = summaries

        return inputs

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, collate_fn = collator)
    predictions = []
    ground_truths = []

    for batch in tqdm(dataloader):  
        inputs = batch["input_ids"].to("cuda:0")
        attention_mask = batch["attention_mask"].to("cuda:0")

        with torch.no_grad():
            outputs = model.generate(
                input_ids=inputs,
                attention_mask=attention_mask,
                max_length=250,
                num_beams=4,
                early_stopping=True,
            )

        decoded_outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True, clean_up_tokenization_spaces=True)

        predictions.extend(decoded_outputs)
        ground_truths.extend(batch["summary"])

    rouge_results = task_evaluator.compute(
        predictions=predictions,
        references=ground_truths,
        use_stemmer=True,
        rouge_types=["rouge1", "rouge2", "rougeL"],
    )

    return rouge_results






if __name__ == "__main__":
    # Load the dataset
    test_ds = load_from_disk("mistral-KD/data/cnn_eval")
    test_ds.set_format("torch", columns=["article", "summary"])

    # Load the model and tokenizer
    teacher_name = "google/gemma-2-9b-it"
    teacher = AutoModelForCausalLM.from_pretrained(
        teacher_name, torch_dtype=torch.float16
    ).to("cuda:0")  

    tokenizer = AutoTokenizer.from_pretrained(teacher_name)

    # Evaluate the model using ROUGE
    teacher.eval()
    rouge_results = evaluate_rouge(teacher, tokenizer, test_ds)

    print(rouge_results)
    with open("mistral-KD/eval_results/teacher_rouge_results.json", "w") as f:
        json.dump(rouge_results, f, indent=4)



