import os
import transformers, torch
from datasets import load_dataset, load_from_disk
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, pipeline
from huggingface_hub import login
from evaluate import evaluator
import json
import torch._dynamo as dynamo


torch.set_float32_matmul_precision("high")
evaluator.logging.enable_progress_bar()  
dynamo.config.cache_size_limit = 64

huggingface_token = os.environ.get('HUGGINGFACE_HUB_TOKEN')
if huggingface_token is None:
    raise ValueError("Please set the HUGGINGFACE_HUB_TOKEN environment variable.")

login(token=huggingface_token)


teacher_name = "google/gemma-2-9b-it"
teacher = AutoModelForCausalLM.from_pretrained(
        teacher_name, torch_dtype=torch.float16
    ).to("cuda:0")  

tokenizer = AutoTokenizer.from_pretrained(teacher_name)
test_ds = load_from_disk("mistral-KD/data/cnn_eval")


task_evaluator = evaluator("text2text-generation")

pipe = pipeline(  
    "text-generation",  
    model=teacher,  
    tokenizer=tokenizer,  
    max_new_tokens=250,  
    do_sample=False  # Gredy
)  

class DecoderOnlyPipeline:  
    def __init__(self, pipeline_obj):  
        self.pipeline = pipeline_obj  
        self.task = "text2text-generation"  # This tells the evaluator what task we're doing  
      
    def __call__(self, inputs, **kwargs):  
        generated_texts = []  
        for text in inputs:  
            # Generate text with Gemma-2  
            outputs = self.pipeline(text, **kwargs)  
            # Extract the generated text (removing the prompt)  
            generated_text = outputs[0]['generated_text'][len(text):].strip()  
            generated_texts.append({"generated_text": generated_text})  
        return generated_texts  


gemma_pipe = DecoderOnlyPipeline(pipe)  



results = task_evaluator.compute(
    model_or_pipeline = gemma_pipe,
    metric = "rouge",
    tokenizer = tokenizer,
    data = test_ds,
    strategy = "bootstrap",
    n_resamples = 250,
    label_column = "summary",
    input_column = "article",
    device = 0,
)



print(results)

with open("eval_results/teacher_rouge_results.json", "w") as f:
    json.dump(results, f, indent=4)


