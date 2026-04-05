#!/usr/bin/env python3
"""Data Pipeline for Knowledge Distillation - GSM8K CoT Trace Generation
Teacher: google/gemma-4-E4B-it (9B, fits on RTX A6000 with bfloat16)
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path("/app/ml_project_0924")
DATA_DIR = PROJECT_ROOT / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

TEACHER_MODEL = "google/gemma-4-E4B-it"

PROMPT_TEMPLATE = """Problem: {question}

Solve step-by-step, showing your reasoning clearly. End with "The answer is: <number>".

Solution:
"""


def load_gsm8k_dataset(split: str = "train", n_samples: Optional[int] = None) -> List[Dict]:
    logger.info(f"Loading GSM8K {split}...")
    dataset = load_dataset("openai/gsm8k", "main", split=split)
    if n_samples:
        dataset = dataset.select(range(min(n_samples, len(dataset))))
    samples = [{"question": item["question"], "answer": item["answer"]} for item in dataset]
    logger.info(f"Loaded {len(samples)} samples")
    return samples


def generate_cot_trace(model, tokenizer, question: str, max_new_tokens: int = 256) -> str:
    prompt = PROMPT_TEMPLATE.format(question=question)
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    inputs = {k: v.to("cuda") for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,  # greedy decoding
            pad_token_id=tokenizer.eos_token_id,
        )
    input_len = inputs["input_ids"].shape[1]
    return tokenizer.decode(outputs[0, input_len:], skip_special_tokens=True).strip()


def main(n_samples: int = 50):
    logger.info(f"Starting CoT generation — teacher: {TEACHER_MODEL}")
    assert torch.cuda.is_available(), "CUDA required"
    logger.info(f"GPU: {torch.cuda.get_device_name(0)}")

    logger.info(f"Loading teacher model: {TEACHER_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(TEACHER_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        TEACHER_MODEL,
        torch_dtype=torch.bfloat16,
        device_map="cuda",  # NO CPU offload
    )
    model.eval()
    logger.info(f"Teacher loaded. GPU memory: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

    samples = load_gsm8k_dataset(split="train", n_samples=n_samples)

    traces = []
    for i, sample in enumerate(samples):
        logger.info(f"Generating trace {i+1}/{len(samples)}")
        try:
            cot = generate_cot_trace(model, tokenizer, sample["question"])
            traces.append({
                "id": i,
                "question": sample["question"],
                "cot_trace": cot,
                "ground_truth": sample["answer"],
                "teacher_model": TEACHER_MODEL,
            })
        except Exception as e:
            logger.error(f"Error on sample {i}: {e}")
            traces.append({
                "id": i,
                "question": sample["question"],
                "cot_trace": "",
                "ground_truth": sample["answer"],
                "error": str(e),
            })

    save_path = DATA_DIR / "cot_traces.json"
    with open(save_path, "w") as f:
        json.dump(traces, f, indent=2)

    valid = sum(1 for t in traces if t.get("cot_trace", "").strip())
    logger.info(f"Saved {len(traces)} traces ({valid} valid) to {save_path}")
    return traces


if __name__ == "__main__":
    main(n_samples=50)