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
# if Dynamo errors, suppress and run in eager mode
torch._dynamo.config.suppress_errors = True


huggingface_token = os.environ.get('HUGGINGFACE_HUB_TOKEN')
if huggingface_token is None:
    raise ValueError("Please set the HUGGINGFACE_HUB_TOKEN environment variable.")

login(token=huggingface_token)




def evaluate_rouge(model, tokenizer, dataset):

    task_evaluator = evaluate.load("rouge")
    task_evaluator_2 = evaluate.load('bertscore')

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

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=18, collate_fn = collator)
    predictions = []
    ground_truths = []

    for batch in tqdm(dataloader):  
        inputs = batch["input_ids"].to("cuda:0")
        attention_mask = batch["attention_mask"].to("cuda:0")

        with torch.no_grad():
            outputs = model.generate(
                input_ids=inputs,
                attention_mask=attention_mask,
                max_new_tokens=250,
                do_sample=False,  # Greedy decoding
                # num_beam = 4 switch to beam search maybe
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

    bertscore_results = task_evaluator_2.compute(
        predictions=predictions,
        references=ground_truths,
        lang="en",
        rescale_with_baseline=True, #When set to True, BERTScore rescales the raw similarity scores using a baseline derived from comparing random sentence pairs. 
        #This helps to make the scores more interpretable by adjusting for the baseline similarity that even unrelated sentences might have.
        use_fast_tokenizer=True,
    )

    return rouge_results, bertscore_results






if __name__ == "__main__":
    # Load the dataset
    test_ds = load_from_disk("mistral-KD/data/cnn_eval")
    # Load the model and tokenizer
    teacher_name = "google/gemma-2-9b-it"
    teacher = AutoModelForCausalLM.from_pretrained(
        teacher_name, torch_dtype=torch.float16, attn_implementation="flash_attention_2"
    ).to("cuda:0")  

    tokenizer = AutoTokenizer.from_pretrained(teacher_name)

    # Evaluate the model using ROUGE
    teacher.eval()
    rouge_results, bertscore_results = evaluate_rouge(teacher, tokenizer, test_ds)

    print(rouge_results)
    print(bertscore_results)
    with open("mistral-KD/eval_results/teacher_rouge_results.json", "w") as f:
        json.dump(rouge_results, f, indent=4)


    with open("mistral-KD/eval_results/teacher_bertscore_results.json", "w") as f:
        json.dump(bertscore_results, f, indent=4)

