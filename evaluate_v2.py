#!/usr/bin/env python3
"""
Evaluation v2 — Student model trained on Qwen3.6-plus CoT traces
Evaluates on GSM8K test set (20 samples), saves metrics to results/
"""

import json
import re
import logging
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path("/app/ml_project_0924")
MODEL_PATH = PROJECT_ROOT / "models" / "distilled_gemma_e2b_v2"
RESULTS_PATH = Path("/root/projects/tasks/02-distillation-31b-e4b/results")
RESULTS_PATH.mkdir(parents=True, exist_ok=True)

STUDENT_BASE = "google/gemma-4-E2B-it"
TEACHER_MODEL = "qwen/qwen3.6-plus:free (OpenRouter)"
N_TEST = 20

PROMPT_TEMPLATE = """Problem: {question}

Solve step-by-step, showing your reasoning clearly. End with "The answer is: <number>".

Solution:
"""


def extract_answer(text: str) -> str:
    match = re.search(r"[Tt]he answer is[:\s]+([0-9,.\-]+)", text)
    if match:
        return match.group(1).replace(",", "").strip()
    numbers = re.findall(r"\b\d+(?:\.\d+)?\b", text)
    return numbers[-1] if numbers else ""


def extract_gt_answer(text: str) -> str:
    match = re.search(r"####\s*([0-9,.\-]+)", text)
    if match:
        return match.group(1).replace(",", "").strip()
    return text.strip().split()[-1]


def simple_bleu(pred: str, ref: str) -> float:
    pred_tokens = set(pred.lower().split())
    ref_tokens = set(ref.lower().split())
    if not ref_tokens:
        return 0.0
    return len(pred_tokens & ref_tokens) / len(ref_tokens)


def main():
    logger.info("=== Evaluation v2 ===")
    assert torch.cuda.is_available(), "CUDA required"

    tokenizer = AutoTokenizer.from_pretrained(STUDENT_BASE)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    logger.info(f"Loading student from {MODEL_PATH}")
    model = AutoModelForCausalLM.from_pretrained(
        str(MODEL_PATH), torch_dtype=torch.bfloat16, device_map="cuda"
    )
    model.eval()

    dataset = load_dataset("openai/gsm8k", "main", split="test")
    test_samples = [{"question": d["question"], "answer": d["answer"]}
                    for d in dataset.select(range(N_TEST))]

    results = []
    correct = 0
    bleu_scores = []

    for i, sample in enumerate(test_samples):
        prompt = PROMPT_TEMPLATE.format(question=sample["question"])
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to("cuda") for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model.generate(
                **inputs, max_new_tokens=256, do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )
        pred_text = tokenizer.decode(outputs[0, inputs["input_ids"].shape[1]:],
                                     skip_special_tokens=True).strip()
        pred_ans = extract_answer(pred_text)
        gt_ans = extract_gt_answer(sample["answer"])
        is_correct = pred_ans == gt_ans
        bleu = simple_bleu(pred_text, sample["answer"])

        correct += int(is_correct)
        bleu_scores.append(bleu)
        results.append({
            "id": i,
            "question": sample["question"],
            "predicted": pred_text,
            "predicted_answer": pred_ans,
            "ground_truth_answer": gt_ans,
            "correct": is_correct,
            "bleu": bleu,
        })
        logger.info(f"[{i+1}/{N_TEST}] correct={is_correct}, pred={pred_ans}, gt={gt_ans}")

    accuracy = correct / N_TEST
    avg_bleu = sum(bleu_scores) / len(bleu_scores)

    metrics = {
        "version": "v2",
        "model": f"{STUDENT_BASE} distilled from {TEACHER_MODEL}",
        "n_test_samples": N_TEST,
        "accuracy": accuracy,
        "correct": correct,
        "avg_bleu": avg_bleu,
        "teacher_model": TEACHER_MODEL,
        "student_model": STUDENT_BASE,
        "training_traces": 500,
        "epochs_completed": 5,
        "distillation_fix": "Proper chat-formatted prompts + trace quality filtering",
        "key_change_from_v1": "Teacher switched from gemma-4-E4B-it (degenerate traces) to qwen3-plus (real CoT reasoning)",
    }

    with open(RESULTS_PATH / "evaluation_metrics_v2.json", "w") as f:
        json.dump(metrics, f, indent=2)
    with open(RESULTS_PATH / "per_sample_results_v2.json", "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"\n{'='*40}")
    logger.info(f"Accuracy: {accuracy:.1%} ({correct}/{N_TEST})")
    logger.info(f"Avg BLEU: {avg_bleu:.3f}")
    logger.info(f"Results saved to {RESULTS_PATH}")
    return metrics


if __name__ == "__main__":
    main()
