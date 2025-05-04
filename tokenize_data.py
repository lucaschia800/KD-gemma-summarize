import os, json, torch
from datasets import load_from_disk, concatenate_datasets
from transformers import AutoTokenizer


teacher_name = "google/gemma-2-9b-it"


tokenizer = AutoTokenizer.from_pretrained(teacher_name)

MAX_LEN = 2048
IGNORE_IDX = -100

def build_example(example):
    """
    Replicates exactly the steps in DataCollatorForChatML.__call__ for one sample.
    Returns the final dict that the collator expects to find inside the dataset.
    """
    full_chat = tokenizer.apply_chat_template(
        example["messages"],
        tokenize=False,
        add_generation_prompt=False,        # same as collator for message path
    )
    full_tok = tokenizer(
        full_chat,
        truncation=True,
        max_length=MAX_LEN,
        padding=False,
        add_special_tokens=False,
    )

    prompt_only = example["messages"][:-1]
    prompt_chat = tokenizer.apply_chat_template(
        prompt_only,
        tokenize=False,
        add_generation_prompt=True,         # collator uses True for prompts
    )
    prompt_tok = tokenizer(
        prompt_chat,
        truncation=True,
        max_length=len(full_tok["input_ids"]),   # same cut‑off rule
        padding=False,
        add_special_tokens=False,
    )

    labels = [IGNORE_IDX] * len(full_tok["input_ids"])
    start = len(prompt_tok["input_ids"])
    labels[start:] = full_tok["input_ids"][start:]

    return {
        # everything the collator normally assembles
        "input_ids"           : full_tok["input_ids"],
        "attention_mask"      : full_tok["attention_mask"],
        "labels"              : labels,
        # keep these so the collator can *skip* heavy work but still build the batch
        "prompt"              : prompt_chat,          # string
        "prompt_input_ids"    : prompt_tok["input_ids"],
        "prompt_attention_mask": prompt_tok["attention_mask"],
        # you may keep original messages if you need them later
        "messages"            : example["messages"],
    }

# -------------------------------------------------------------------------
#  Load raw datasets, concatenate, and preprocess                        --
# -------------------------------------------------------------------------
xsum = load_from_disk("mistral-KD/data/xsum_formatted")
cnn  = load_from_disk("mistral-KD/data/cnn_formatted")
sci1 = load_from_disk("mistral-KD/data/sci1_formatted")

raw_ds = concatenate_datasets([xsum, cnn, sci1])

print(raw_ds[0])

proc_ds = raw_ds.map(
    build_example,
    num_proc=16,          
    remove_columns=raw_ds.column_names,  # drop old columns, keep the new ones
    batch
)

print(proc_ds[0])
print(tokenizer.decode(proc_ds[0]["input_ids"]))

proc_ds.save_to_disk("mistral-KD/data/chatml_tokenised")
print(" Saved to mistral-KD/data/chatml_tokenised")