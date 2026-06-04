import torch
from unsloth import FastLanguageModel
from trl import SFTTrainer, SFTConfig
from datasets import load_from_disk
import wandb

MODEL_ID = r"C:\Users\ARYAN\.cache\huggingface\hub\models--meta-llama--Llama-3.2-3B-Instruct\snapshots\0cb88a4f764b7a12671c53f0838cd831a0843b95"
MAX_LEN = 1024

wandb.init(project="huggingface", name="llama32-3b-python-qlora")

print("Loading model with unsloth...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_ID,
    max_seq_length=MAX_LEN,
    dtype=torch.float16,
    load_in_4bit=True,
    token=True,
)

print("Attaching LoRA adapters...")
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=42,
)

print("Loading dataset...")
dataset = load_from_disk("./data/formatted_dataset")
train_dataset = dataset["train"].select_columns(["text"])
eval_dataset  = dataset["test"].select_columns(["text"])

training_args = SFTConfig(
    output_dir="./checkpoints",
    num_train_epochs=3,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    warmup_ratio=0.05,
    fp16=True,
    logging_steps=10,
    eval_strategy="steps",
    eval_steps=200,
    save_strategy="steps",
    save_steps=200,
    save_total_limit=3,
    load_best_model_at_end=True,
    report_to="wandb",
    run_name="llama32-3b-python-qlora",
    dataset_text_field="text",
    max_seq_length=MAX_LEN,
)

trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    processing_class=tokenizer,
    args=training_args,
)

print("Starting training...")
trainer.train()

model.save_pretrained("./lora-adapter")
tokenizer.save_pretrained("./lora-adapter")
print("Adapter saved to ./lora-adapter")
wandb.finish()