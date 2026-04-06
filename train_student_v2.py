#!/usr/bin/env python3
"""
Student Training v2 — gemma-4-E2B-it on Qwen3.6-plus CoT traces
Uses proper chat-formatted training data. 5 epochs, save_total_limit=1 (no disk issues).
"""

import json
import logging
import os
from pathlib import Path
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from torch.utils.data import Dataset

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path("/app/ml_project_0924")
DATA_PATH = PROJECT_ROOT / "data" / "cot_traces_v2.json"
MODEL_SAVE_PATH = PROJECT_ROOT / "models" / "distilled_gemma_e2b_v2"
HF_EXPORTS_PATH = PROJECT_ROOT / "hf_exports" / "gemma-e2b-distilled-v2"

STUDENT_MODEL = "google/gemma-4-E2B-it"


class CoTDataset(Dataset):
    def __init__(self, traces, tokenizer, max_length=384):
        self.items = []
        skipped = 0
        for trace in traces:
            if not trace.get("valid", True):
                skipped += 1
                continue
            cot = trace.get("cot_trace", "").strip()
            if not cot or len(cot.split()) < 25:
                skipped += 1
                continue
            # Format as instruction-following text matching student model expectations
            text = f"Problem: {trace['question']}\n\nSolution:\n{cot}"
            enc = tokenizer(text, truncation=True, max_length=max_length, return_tensors="pt")
            self.items.append({
                "input_ids": enc["input_ids"][0],
                "attention_mask": enc["attention_mask"][0],
                "labels": enc["input_ids"][0].clone(),
            })
        logger.info(f"Dataset: {len(self.items)} samples (skipped {skipped} invalid)")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]


def main():
    logger.info("=== Knowledge Distillation Training v2 (Qwen3.6-plus traces) ===")
    assert torch.cuda.is_available(), "CUDA required"
    logger.info(f"GPU: {torch.cuda.get_device_name(0)}, Free: {torch.cuda.mem_get_info()[0]/1e9:.1f} GB")

    with open(DATA_PATH) as f:
        traces = json.load(f)
    logger.info(f"Loaded {len(traces)} traces from {DATA_PATH}")

    tokenizer = AutoTokenizer.from_pretrained(STUDENT_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    logger.info(f"Loading student: {STUDENT_MODEL}")
    student = AutoModelForCausalLM.from_pretrained(
        STUDENT_MODEL,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
    )
    student.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    student.enable_input_require_grads()

    dataset = CoTDataset(traces, tokenizer, max_length=384)
    logger.info(f"Training on {len(dataset)} valid samples")

    MODEL_SAVE_PATH.mkdir(parents=True, exist_ok=True)
    training_args = TrainingArguments(
        output_dir=str(MODEL_SAVE_PATH),
        num_train_epochs=5,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=2e-5,
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=1,       # keep only best checkpoint — no disk explosion
        report_to="none",
        remove_unused_columns=False,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="adamw_torch_fused",
        max_grad_norm=1.0,
        dataloader_num_workers=0,
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    trainer = Trainer(
        model=student,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
    )

    logger.info("Training started...")
    trainer.train()
    logger.info("Training complete.")

    trainer.save_model(str(MODEL_SAVE_PATH))
    tokenizer.save_pretrained(str(MODEL_SAVE_PATH))

    HF_EXPORTS_PATH.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(HF_EXPORTS_PATH))
    tokenizer.save_pretrained(str(HF_EXPORTS_PATH))
    logger.info(f"Saved to {MODEL_SAVE_PATH} and {HF_EXPORTS_PATH}")


if __name__ == "__main__":
    main()
