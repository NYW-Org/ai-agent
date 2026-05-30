import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


class QwenInferenceEngine:
    def __init__(self, model_id: str, cache_dir: str):
        self.model_id = model_id
        self.cache_dir = cache_dir
        self.tokenizer = None
        self.model = None
        self.hf_token="xxxxxx"
        self.base_model_id = "Qwen/Qwen2.5-7B-Instruct"

    def initialize(self):
        print(f"🔄 Initializing Hardware LLM Tensors: {self.model_id}...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_id,
            cache_dir=self.cache_dir,
            trust_remote_code=True,
            token=self.hf_token
        )
        base_model = AutoModelForCausalLM.from_pretrained(
            self.base_model_id,
            cache_dir=self.cache_dir,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
            token=self.hf_token
        )

        print(f"🪄 Merging Fine-Tuned LoRA Adapters from: {self.model_id}...")
        try:
            # 3. Dynamic runtime patching overlays your custom parameters
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(
                base_model,
                self.model_id,
                token=self.hf_token
            )
            print("✅ Fine-tuned adapters successfully fused with Base Model.")
        except ImportError:
            # Fallback if peft wasn't explicitly pinned in the environment layer yet
            print("⚠️ PEFT library missing. Using base model processing fallback layer.")
            self.model = base_model

    def generate_text(self, formatted_prompt: str) -> str:
        if next(self.model.parameters()).device.type != "cuda":
            print("🚀 GPU Worker detected! Moving model tensors to CUDA VRAM...")
            self.model = self.model.to("cuda")

        inputs = self.tokenizer([formatted_prompt], return_tensors="pt").to("cuda")
        with torch.inference_mode():
            generated_ids = self.model.generate(**inputs, max_new_tokens=256, temperature=0.7, do_sample=True)

        generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, generated_ids)]
        return self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
