"""
======================================================================
SCIQA_PROCESS ---

ScienceQA 多模态验证脚本

功能:
1. 加载 LLaVA/LLaVA-Next 学生模型（可选 LoRA 适配器）
2. 使用 ScienceQA 的图像+文本样本进行推理
3. 解析模型输出并计算选择题准确率

Author: VQ-LoRD Project
Created: January 2026
======================================================================
"""

import argparse
import json
import os
import re
from typing import List, Optional, Tuple

import torch
from datasets import load_dataset
from transformers import LlavaNextForConditionalGeneration, LlavaNextProcessor, BitsAndBytesConfig
from peft import PeftModel
from tqdm import tqdm


def build_prompt(question: str, choices: List[str]) -> str:
    choices_text = ""
    for idx, choice in enumerate(choices):
        choices_text += f"({chr(65 + idx)}) {choice}\n"
    return f"<image>\nQuestion: {question}\nOptions:\n{choices_text}Answer:"


def normalize_text(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text


def extract_choice_from_output(output_text: str, choices: List[str]) -> Optional[int]:
    """尽量从输出中抽取选择题答案索引。"""
    if not output_text:
        return None

    normalized = normalize_text(output_text)

    match = re.search(r"answer\s*[:：]\s*([a-z])", normalized)
    if match:
        letter = match.group(1).upper()
        idx = ord(letter) - 65
        if 0 <= idx < len(choices):
            return idx

    match = re.search(r"\b([a-z])\b", normalized)
    if match:
        letter = match.group(1).upper()
        idx = ord(letter) - 65
        if 0 <= idx < len(choices):
            return idx

    choice_hits = []
    for idx, choice in enumerate(choices):
        if not choice:
            continue
        if normalize_text(choice) in normalized:
            choice_hits.append(idx)

    if len(choice_hits) == 1:
        return choice_hits[0]

    return None


def load_model_and_processor(model_path: str, adapter_path: str, use_4bit: int):
    if use_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
        )
        model = LlavaNextForConditionalGeneration.from_pretrained(
            model_path,
            device_map="auto",
            quantization_config=quantization_config,
            trust_remote_code=True,
        )
    else:
        model = LlavaNextForConditionalGeneration.from_pretrained(
            model_path,
            device_map="auto",
            torch_dtype=torch.float16,
            trust_remote_code=True,
        )

    if adapter_path and os.path.exists(adapter_path):
        model = PeftModel.from_pretrained(model, adapter_path)
        model.eval()

    processor = LlavaNextProcessor.from_pretrained(model_path, trust_remote_code=True)
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token

    return model, processor


def run_eval(
    model,
    processor,
    split: str,
    max_samples: int,
    max_new_tokens: int,
    save_path: str,
):
    dataset = load_dataset("derek-thomas/ScienceQA", split=split)
    dataset_with_images = [item for item in dataset if item.get("image") is not None]

    if max_samples > 0 and len(dataset_with_images) > max_samples:
        dataset_with_images = dataset_with_images[:max_samples]

    results = []
    correct = 0
    total = 0

    for item in tqdm(dataset_with_images, desc="ScienceQA Eval"):
        question = item.get("question", "")
        choices = item.get("choices", [])
        answer_idx = item.get("answer", 0)
        image = item.get("image")

        prompt = build_prompt(question, choices)
        inputs = processor(
            text=prompt,
            images=image,
            return_tensors="pt",
            padding="longest",
            truncation=False,
        )

        input_ids = inputs["input_ids"].to(model.device)
        attention_mask = inputs["attention_mask"].to(model.device)
        pixel_values = inputs["pixel_values"].to(model.device)
        image_sizes = inputs.get("image_sizes")
        if image_sizes is not None:
            image_sizes = image_sizes.to(model.device)

        with torch.no_grad():
            generated = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                image_sizes=image_sizes,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=model.config.pad_token_id or processor.tokenizer.pad_token_id,
            )

        output_text = processor.tokenizer.decode(generated[0], skip_special_tokens=True)
        pred_idx = extract_choice_from_output(output_text, choices)
        is_correct = pred_idx == answer_idx

        total += 1
        if is_correct:
            correct += 1

        results.append({
            "question": question,
            "choices": choices,
            "answer_idx": answer_idx,
            "pred_idx": pred_idx,
            "output": output_text,
            "correct": is_correct,
        })

    accuracy = correct / total if total else 0.0
    metrics = {"accuracy": accuracy, "total": total, "correct": correct}

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump({"metrics": metrics, "results": results}, f, ensure_ascii=False, indent=2)

    print(f"Accuracy: {accuracy:.4f} ({correct}/{total})")
    print(f"Results saved to: {save_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="ScienceQA 多模态验证脚本")
    parser.add_argument("--model_path", type=str, required=True, help="基础模型路径")
    parser.add_argument("--adapter_path", type=str, default="", help="LoRA 适配器路径")
    parser.add_argument("--split", type=str, default="validation", help="ScienceQA split")
    parser.add_argument("--max_samples", type=int, default=200, help="最大评测样本数")
    parser.add_argument("--max_new_tokens", type=int, default=64, help="生成长度")
    parser.add_argument("--use_4bit", type=int, default=1, help="是否使用4bit加载")
    parser.add_argument("--save_path", type=str, default="./sciqa_eval.json", help="保存结果路径")
    return parser.parse_args()


def main():
    args = parse_args()
    model, processor = load_model_and_processor(
        args.model_path, args.adapter_path, args.use_4bit
    )
    run_eval(
        model=model,
        processor=processor,
        split=args.split,
        max_samples=args.max_samples,
        max_new_tokens=args.max_new_tokens,
        save_path=args.save_path,
    )


if __name__ == "__main__":
    main()
