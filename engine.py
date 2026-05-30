import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


class QwenInferenceEngine:
    def __init__(self, model_id: str, cache_dir: str):
        self.model_id = model_id
        self.cache_dir = cache_dir
        self.tokenizer = None
        self.model = None

    def initialize(self):
        print(f"🔄 Initializing Hardware LLM Tensors: {self.model_id}...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_id,
            cache_dir=self.cache_dir,
            trust_remote_code=True,
            token=self.hf_token
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            cache_dir=self.cache_dir,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
            token=self.hf_token
        )
        print("✅ Qwen model engine active in VRAM.")

    def generate_text(self, formatted_prompt: str) -> str:
        inputs = self.tokenizer([formatted_prompt], return_tensors="pt").to("cuda")
        with torch.inference_mode():
            generated_ids = self.model.generate(**inputs, max_new_tokens=256, temperature=0.7, do_sample=True)

        generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, generated_ids)]
        return self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
