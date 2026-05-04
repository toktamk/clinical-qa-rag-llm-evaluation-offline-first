from transformers import AutoModelForCausalLM, AutoTokenizer
import torch


class LocalLLM:
    def __init__(self, model_name: str = "mistralai/Mistral-7B-Instruct-v0.2"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )

    def generate(self, prompt: str, max_tokens: int = 512) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=0.2,
            do_sample=False
        )

        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)