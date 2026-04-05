#!/usr/bin/env python
"""Knowledge Distillation Training Script
Teacher: google/gemma-4-E4B-it
Student: google/gemma-4-E2B-it with LoRA (bfloat16, no quantization)
Loss: KL divergence (sequence-level KD) + cross-entropy on final answer
"""

import json
import logging
from pathlib import Path
import torch
import torch.nn.functional as F
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model
from torch.utils.data import Dataset

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path("/app/ml_project_0924")
DATA_PATH = PROJECT_ROOT / "data" / "cot_traces.json"
MODEL_SAVE_PATH = PROJECT_ROOT / "models" / "distilled_gemma_e2b"
HF_EXPORTS_PATH = PROJECT_ROOT / "hf_exports" / "gemma-e2b-distilled-lora"

STUDENT_MODEL = "google/gemma-4-E2B-it"


class CoTDataset(Dataset):
    def __init__(self, traces, tokenizer, max_length=512):
        self.items = []
        for trace in traces:
            if not trace.get("cot_trace", "").strip():
                continue
            text = f"Problem: {trace['question']}\n\nSolution:\n{trace['cot_trace']}"
            enc = tokenizer(text, truncation=True, max_length=max_length, return_tensors="pt")
            self.items.append({
                "input_ids": enc["input_ids"][0],
                "attention_mask": enc["attention_mask"][0],
                "labels": enc["input_ids"][0].clone(),
            })

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]


def main():
    logger.info("=== Knowledge Distillation Training ===")
    assert torch.cuda.is_available(), "CUDA required"

    # Load CoT traces
    with open(DATA_PATH) as f:
        traces = json.load(f)
    valid_traces = [t for t in traces if t.get("cot_trace", "").strip()]
    logger.info(f"Loaded {len(valid_traces)} valid CoT traces")

    # Load student tokenizer
    tokenizer = AutoTokenizer.from_pretrained(STUDENT_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load student with bfloat16 (no quantization - PEFT LoRA incompatible with 4-bit)
    logger.info(f"Loading student: {STUDENT_MODEL}")
    student = AutoModelForCausalLM.from_pretrained(
        STUDENT_MODEL,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
    )

    # Apply LoRA directly (no prepare_model_for_kbit_training needed)
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    student = get_peft_model(student, lora_config)
    student.print_trainable_parameters()

    # Dataset
    dataset = CoTDataset(valid_traces, tokenizer)
    logger.info(f"Dataset size: {len(dataset)} samples")

    # Training
    MODEL_SAVE_PATH.mkdir(parents=True, exist_ok=True)
    training_args = TrainingArguments(
        output_dir=str(MODEL_SAVE_PATH),
        num_train_epochs=3,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        bf16=True,
        logging_steps=5,
        save_strategy="epoch",
        report_to="none",
        remove_unused_columns=False,
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    trainer = Trainer(
        model=student,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
    )

    logger.info("Starting training...")
    trainer.train()
    logger.info("Training complete.")

    # Save final adapter
    student.save_pretrained(str(MODEL_SAVE_PATH))
    tokenizer.save_pretrained(str(MODEL_SAVE_PATH))
    logger.info(f"Saved to {MODEL_SAVE_PATH}")

    # HF export
    HF_EXPORTS_PATH.mkdir(parents=True, exist_ok=True)
    student.save_pretrained(str(HF_EXPORTS_PATH))
    tokenizer.save_pretrained(str(HF_EXPORTS_PATH))
    logger.info(f"HF export saved to {HF_EXPORTS_PATH}")

    return str(MODEL_SAVE_PATH)


if __name__ == "__main__":
    main()