import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE_MODEL = r"C:\Users\ARYAN\.cache\huggingface\hub\models--meta-llama--Llama-3.2-3B-Instruct\snapshots\0cb88a4f764b7a12671c53f0838cd831a0843b95"
ADAPTER_DIR = "./lora-adapter"
MERGED_DIR = "./merged-model"

print("Loading base model in fp16 on CPU...")
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float16,
    device_map="cpu",
)

print("Loading LoRA adapter...")
model = PeftModel.from_pretrained(model, ADAPTER_DIR)

print("Merging weights...")
model = model.merge_and_unload()

print("Saving merged model...")
model.save_pretrained(MERGED_DIR, safe_serialization=True)

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
tokenizer.save_pretrained(MERGED_DIR)

print("Done. Merged model saved to ./merged-model")