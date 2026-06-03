from datasets import load_dataset
from transformers import AutoTokenizer

MODEL_ID = "meta-llama/Llama-3.2-3B-Instruct"

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

print("Downloading dataset...")
dataset = load_dataset("iamtarun/python_code_instructions_18k_alpaca", split="train")
print(f"Raw dataset size: {len(dataset)} samples")

print("\n--- RAW SAMPLE EXAMPLE ---")
print("instruction:", dataset[0]["instruction"])
print("input:", dataset[0]["input"])
print("output:", dataset[0]["output"][:200])
print("--------------------------\n")

def format_sample(row):
    instruction = row["instruction"]
    context = row["input"] or ""
    response = row["output"]

    messages = [
        {
            "role": "system",
            "content": "You are an expert Python programmer. Write clean, correct, well-commented Python code."
        },
        {
            "role": "user",
            "content": instruction + (f"\n\nContext:\n{context}" if context else "")
        },
        {
            "role": "assistant",
            "content": response
        },
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False
    )
    return {"text": text}

print("Formatting samples into Llama 3.2 chat template...")
dataset = dataset.map(format_sample, desc="Formatting")

print("\n--- FORMATTED SAMPLE EXAMPLE ---")
print(dataset[0]["text"][:500])
print("--------------------------------\n")

before = len(dataset)
dataset = dataset.filter(lambda x: len(x["text"]) < 3000, desc="Filtering long samples")
after = len(dataset)
print(f"Filtered {before - after} long samples. Remaining: {after}")

dataset = dataset.train_test_split(test_size=0.1, seed=42)

print(f"\nFinal split:")
print(f"  Train: {len(dataset['train'])} samples")
print(f"  Validation: {len(dataset['test'])} samples")

dataset.save_to_disk("./data/formatted_dataset")
print("\nDataset saved to ./data/formatted_dataset")
print("Done.")