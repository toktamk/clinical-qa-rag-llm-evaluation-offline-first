from __future__ import annotations

from dataclasses import dataclass

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass(frozen=True)
class LocalLLMConfig:
    model_name: str = "Qwen/Qwen2.5-3B-Instruct"
    device: str = "cpu"
    max_new_tokens: int = 512
    temperature: float = 0.0
    do_sample: bool = False


class LocalLLM:
    """
    Thin local LLM wrapper for offline generation.

    First run may download model weights from Hugging Face.
    After the model is cached locally, this can run offline.
    """

    def __init__(self, config: LocalLLMConfig):
        self.config = config

        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)

        torch_dtype = torch.float32
        if config.device == "cuda":
            torch_dtype = torch.float16

        self.model = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            torch_dtype=torch_dtype,
        )

        self.model.to(config.device)
        self.model.eval()

    def generate(
        self,
        prompt: str,
        *,
        max_new_tokens: int | None = None,
        temperature: float | None = None,
        do_sample: bool | None = None,
    ) -> str:
        max_new_tokens = max_new_tokens or self.config.max_new_tokens
        temperature = self.config.temperature if temperature is None else temperature
        do_sample = self.config.do_sample if do_sample is None else do_sample

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.config.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature if do_sample else None,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        decoded = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Remove prompt prefix if present.
        if decoded.startswith(prompt):
            return decoded[len(prompt):].strip()

        return decoded.strip()


class MockLLM:
    """
    Deterministic test double for pipeline debugging without loading a model.
    """

    def generate(self, prompt: str, **kwargs) -> str:
        return (
            "Answer: The provided evidence is insufficient to answer this question.\n"
            "Citations: null\n"
            "Abstained: true\n"
            "Abstention reason: insufficient_evidence"
        )