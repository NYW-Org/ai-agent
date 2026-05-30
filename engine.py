import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


class QwenInferenceEngine:
    def __init__(self, model_id: str, cache_dir: str):
        self.model_id = model_id
        self.cache_dir = cache_dir
        self.tokenizer = None
        self.model = None
        self.base_model_id = "Qwen/Qwen2.5-7B-Instruct"
        self.hf_token = "xxxxxx"

    def initialize(self):
        print(f"🚀 Initializing Directly on GPU: Loading Base {self.base_model_id}...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.base_model_id,
            cache_dir=self.cache_dir,
            trust_remote_code=True,
            token=self.hf_token
        )

        # 1. Load the base model directly into CUDA VRAM using device_map="auto"
        base_model = AutoModelForCausalLM.from_pretrained(
            self.base_model_id,
            cache_dir=self.cache_dir,
            torch_dtype=torch.float16,
            device_map="auto",  # 👈 Maps weights straight to GPU VRAM immediately
            trust_remote_code=True,
            token=self.hf_token
        )

        print(f"🪄 Fusing Fine-Tuned LoRA Adapters into VRAM: {self.model_id}...")
        from peft import PeftModel
        # 2. Merge the adapters straight into the live GPU memory space
        fused_model = PeftModel.from_pretrained(
            base_model,
            self.model_id,
            token=self.hf_token
        )

        # 3. Lock it down on the GPU device permanently
        self.model = fused_model.to("cuda")
        print("✅ Production Model Fully Baked into GPU VRAM and ready for instant hits!")

    def generate_text(self, formatted_prompt: str) -> str:
        # ⚡ NO MORE RUNTIME VRAM MIGRATION CODES HERE!
        # The model is already warm and waiting inside the GPU.

        inputs = self.tokenizer([formatted_prompt], return_tensors="pt").to("cuda")
        with torch.inference_mode():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=64,
                # 💡 OPTIMIZATION: Lower this if you only need short responses. Fewer tokens = faster speeds!
                temperature=0.7,
                do_sample=True
            )

        generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, generated_ids)]
        return self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
