# Code LLM — Fine-Tuned Llama 3.2 3B for Python Generation

Fine-tuned Llama 3.2 3B on 16,430 Python instruction pairs using QLoRA.
Converted to GGUF and served locally via Ollama. No cloud API required.

## Results

| Model | HumanEval pass@1 | Tok/sec | RAM | Size |
|---|---|---|---|---|
| llama3.2:3b (base) | ~28% | ~18 | 2.5 GB | 2.0 GB |
| codellm-Q4_K_M (fine-tuned) | TBD | TBD | TBD | TBD |
| codellm-Q8_0 (fine-tuned) | TBD | TBD | TBD | TBD |

*Results will be updated after training and evaluation are complete.*

## Hardware

- GPU: NVIDIA GeForce RTX 5050 Laptop GPU (8 GB VRAM)
- RAM: 16 GB
- OS: Windows 11

## Stack

- **Training:** PyTorch + HuggingFace Transformers + PEFT + TRL
- **Quantization:** optimum-quanto (4-bit QLoRA)
- **Export:** llama.cpp GGUF conversion
- **Serving:** Ollama + FastAPI
- **Eval:** HumanEval pass@1

## Pipeline

- Dataset → Format → QLoRA Fine-tune → Merge → GGUF → Eval → Serve

## Training

- Base model: `meta-llama/Llama-3.2-3B-Instruct`
- Dataset: `iamtarun/python_code_instructions_18k_alpaca` (18,612 samples)
- After filtering: 16,430 train / 1,826 validation
- Method: QLoRA — 4-bit base model + LoRA adapters (r=16, alpha=32)
- Trainable parameters: ~9.2M / 3.2B total (0.28%)
- Epochs: 3
- Effective batch size: 16 (batch=2, grad_accum=8)
- Learning rate: 2e-4 with cosine scheduler

## Training Curves

*W&B training loss curve screenshot will go here*

## Reproduce

```bash
git clone https://github.com/yourusername/codellm
cd codellm

conda create -n codellm python=3.11 -y
conda activate codellm

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install transformers==4.47.0 datasets==3.2.0 accelerate==1.2.1
pip install peft==0.14.0 trl==0.13.0 optimum-quanto wandb

python dataset.py
python train.py
```

## Project Structure

codellm/
├── dataset.py        # Download + format dataset
├── train.py          # QLoRA fine-tuning
├── merge.py          # Merge LoRA adapter into base model
├── convert.py        # Export to GGUF + quantize
├── eval.py           # HumanEval benchmark
├── api_server.py     # FastAPI serving
└── data/             # Formatted dataset (gitignored)

## What I Learned

- How QLoRA works: 4-bit quantization + LoRA adapters train only 0.28% of parameters
- Why chat templates matter: wrong format = model ignores instruction structure
- Quantization tradeoffs: Q4_K_M gives 95% quality at 31% of fp16 file size
- How to evaluate LLMs objectively: HumanEval pass@1 measures real code correctness

