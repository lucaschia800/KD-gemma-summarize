import torch
from datasets import load_from_disk, Dataset # Dataset for creating a dummy raw sample
from transformers import AutoTokenizer

# --- Configuration ---
TOKENIZER_NAME = "google/gemma-2-9b-it" # Or "google/gemma-2-2b-it" for your student model
EXISTING_DATASET_PATH = "/gscratch/stf/lbc800/mistral-KD/data/chatml_tokenised" # Your current tokenized data
MAX_LEN = 1500
IGNORE_IDX = -100

print(f"Loading tokenizer: {TOKENIZER_NAME}")
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
bos_token_id = tokenizer.bos_token_id
bos_token_string = tokenizer.bos_token

print(f"Gemma BOS Token: '{bos_token_string}', ID: {bos_token_id}")
if bos_token_id is None:
    print("CRITICAL WARNING: Tokenizer does not have a bos_token_id. This is highly unexpected for Gemma.")
    exit()

# --- 2. Inspect one sample from  chatml_tokenised dataset ---
print(f"\n--- Part 1: Inspecting one sample from your existing dataset: {EXISTING_DATASET_PATH} ---")
try:
    current_processed_dataset = load_from_disk(EXISTING_DATASET_PATH)
    if len(current_processed_dataset) > 0:
        sample_from_disk = current_processed_dataset[0] # Load the first sample
        input_ids_from_disk = sample_from_disk["input_ids"]
        print(f"Sample 0 from disk - First 10 input_ids: {input_ids_from_disk[:10]}")
        if input_ids_from_disk and input_ids_from_disk[0] == bos_token_id:
            print(f"  VERDICT (Existing Data): Starts with BOS token ID ({bos_token_id}). Looks OK.")
        elif not input_ids_from_disk:
            print(f"  VERDICT (Existing Data): input_ids is empty.")
        else:
            print(f"  VERDICT (Existing Data): DOES NOT start with BOS token ID ({bos_token_id}). First ID: {input_ids_from_disk[0]}. POTENTIAL PROBLEM.")
        print(f"  Decoded (first 50 chars): '{tokenizer.decode(input_ids_from_disk[:50])}'")
    else:
        print("  Your existing dataset is empty.")
except Exception as e:
    print(f"  Could not load or inspect current dataset: {e}")

# --- 3. Test build_example function behavior ---
print("\n--- Part 2: Testing your build_example function's behavior ---")

# Your original build_example function (copied from your script)
def build_example_original(example_data):
    full_chat = tokenizer.apply_chat_template(
        example_data["messages"], tokenize=False, add_generation_prompt=False,
    )
    full_tok = tokenizer(
        full_chat, truncation=True, max_length=MAX_LEN, padding=False, add_special_tokens=False, # Original
    )
    # Simplified for this test, only returning input_ids
    return {"input_ids": full_tok["input_ids"]}

# Modified build_example function with add_special_tokens=True
def build_example_corrected(example_data):
    full_chat = tokenizer.apply_chat_template(
        example_data["messages"], tokenize=False, add_generation_prompt=False,
    )
    full_tok = tokenizer(
        full_chat, truncation=True, max_length=MAX_LEN, padding=False, add_special_tokens=True, # Corrected
    )
    # Simplified for this test, only returning input_ids
    return {"input_ids": full_tok["input_ids"]}

# Create a sample raw input (like one from xsum, cnn, etc. before tokenization)
# This should be in the format build_example function expects
raw_sample_messages = {
    "messages": [
        {"role": "user", "content": "What is the weather like in Seattle today?"},
        {"role": "assistant", "content": "I am an AI and don't have real-time weather information."}
    ]
}
print(f"Using raw sample messages: {raw_sample_messages['messages']}")

# Test with original function (add_special_tokens=False)
print("\nTesting with ORIGINAL build_example (add_special_tokens=False):")
original_processed = build_example_original(raw_sample_messages.copy())
original_input_ids = original_processed["input_ids"]
print(f"  First 10 input_ids: {original_input_ids[:10]}")
if original_input_ids and original_input_ids[0] == bos_token_id:
    print(f"  VERDICT (Original Function): Starts with BOS token ID ({bos_token_id}).")
    print(f"    This implies apply_chat_template included '{bos_token_string}' and it was tokenized.")
else:
    print(f"  VERDICT (Original Function): DOES NOT start with BOS ID ({bos_token_id}). First ID: {original_input_ids[0] if original_input_ids else 'N/A'}. LIKELY PROBLEM SOURCE.")
print(f"  Decoded (first 50 chars): '{tokenizer.decode(original_input_ids[:50])}'")


# Test with corrected function (add_special_tokens=True)
print("\nTesting with CORRECTED build_example (add_special_tokens=True):")
corrected_processed = build_example_corrected(raw_sample_messages.copy())
corrected_input_ids = corrected_processed["input_ids"]
print(f"  First 10 input_ids: {corrected_input_ids[:10]}")
if corrected_input_ids and corrected_input_ids[0] == bos_token_id:
    print(f"  VERDICT (Corrected Function): Starts with BOS token ID ({bos_token_id}). This is expected behavior.")
else:
    print(f"  VERDICT (Corrected Function): DOES NOT start with BOS ID ({bos_token_id}). First ID: {corrected_input_ids[0] if corrected_input_ids else 'N/A'}. (This would be very unusual if tokenizer is standard Gemma).")
print(f"  Decoded (first 50 chars): '{tokenizer.decode(corrected_input_ids[:50])}'")

# --- Optional: Check apply_chat_template output directly ---
print("\n--- Optional: Direct check of apply_chat_template string output ---")
template_string_full = tokenizer.apply_chat_template(
    raw_sample_messages["messages"], tokenize=False, add_generation_prompt=False
)
print(f"String from apply_chat_template (first 50 chars): '{template_string_full[:50]}'")
if template_string_full.startswith(bos_token_string):
    print(f"  INFO: The template string ITSELF starts with '{bos_token_string}'.")
else:
    print(f"  INFO: The template string ITSELF DOES NOT start with '{bos_token_string}'.")