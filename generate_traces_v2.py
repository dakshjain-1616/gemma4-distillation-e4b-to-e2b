#!/usr/bin/env python3
"""
CoT Trace Generation v2 — Teacher: qwen/qwen3.6-plus via OpenRouter API
Fix: proper chat formatting, trace filtering, real reasoning validation
"""

import json
import time
import re
import logging
from pathlib import Path
from openai import OpenAI

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

OPENROUTER_API_KEY = "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"
TEACHER_MODEL = "qwen/qwen3.6-plus:free"
OUTPUT_PATH = Path("/app/ml_project_0924/data/cot_traces_v2.json")
N_SAMPLES = 200  # enough for solid training signal

SYSTEM_PROMPT = """You are a precise math reasoning assistant. When given a word problem, solve it step-by-step showing all arithmetic clearly. Always end your response with exactly: "The answer is: <number>" where <number> is only digits."""

FEW_SHOT = """Example 1:
Problem: Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?

Solution:
Janet's ducks lay 16 eggs per day.
She uses 3 + 4 = 7 eggs herself.
Remaining eggs: 16 - 7 = 9 eggs.
She earns 9 × $2 = $18.
The answer is: 18

Example 2:
Problem: A robe takes 2 bolts of blue fiber and half that much white fiber. How many bolts in total does it take?

Solution:
Blue fiber: 2 bolts.
White fiber: 2 / 2 = 1 bolt.
Total: 2 + 1 = 3 bolts.
The answer is: 3

Now solve the following problem using the same step-by-step format:
"""


def is_valid_trace(trace: str) -> bool:
    """Filter out degenerate traces."""
    if not trace or len(trace.split()) < 25:
        return False
    # Must have a numeric answer at the end
    if not re.search(r"[Tt]he answer is[:\s]+[\d,.\-]+", trace):
        return False
    # Must not be repetitive
    lines = [l.strip() for l in trace.split('\n') if l.strip()]
    if len(lines) >= 3:
        repeated = sum(1 for i in range(1, len(lines)) if lines[i] == lines[i-1])
        if repeated > len(lines) * 0.3:  # >30% repeated lines
            return False
    return True


def generate_trace(client: OpenAI, question: str, retries: int = 6) -> str:
    prompt = FEW_SHOT + f"Problem: {question}\n\nSolution:"
    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model=TEACHER_MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=512,
                temperature=0.0,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            wait = min(30, 4 ** attempt)  # 1s, 4s, 16s, 30s, 30s, 30s
            logger.warning(f"Attempt {attempt+1} failed, retrying in {wait}s: {e}")
            time.sleep(wait)
    return ""


def main():
    from datasets import load_dataset
    logger.info(f"Teacher: {TEACHER_MODEL} via OpenRouter")

    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=OPENROUTER_API_KEY,
    )

    dataset = load_dataset("openai/gsm8k", "main", split="train")
    samples = [{"question": d["question"], "answer": d["answer"]}
               for d in dataset.select(range(N_SAMPLES))]
    logger.info(f"Loaded {len(samples)} GSM8K samples")

    traces = []
    valid_count = 0
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    for i, sample in enumerate(samples):
        trace = generate_trace(client, sample["question"])
        valid = is_valid_trace(trace)
        if valid:
            valid_count += 1
        traces.append({
            "id": i,
            "question": sample["question"],
            "cot_trace": trace,
            "ground_truth": sample["answer"],
            "teacher_model": TEACHER_MODEL,
            "valid": valid,
        })
        # Save incrementally every 10 traces
        if (i + 1) % 10 == 0:
            with open(OUTPUT_PATH, "w") as f:
                json.dump(traces, f, indent=2)
            logger.info(f"Progress: {i+1}/{N_SAMPLES} | Valid: {valid_count} | Saved.")
        time.sleep(1.0)  # rate limit headroom

    with open(OUTPUT_PATH, "w") as f:
        json.dump(traces, f, indent=2)

    logger.info(f"Done. {len(traces)} traces, {valid_count} valid ({valid_count/len(traces)*100:.1f}%) → {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
