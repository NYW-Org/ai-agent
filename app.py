import modal
from fastapi import FastAPI

app = modal.App(name="qwen-session-api")
hf_volume = modal.Volume.from_name("qwen-model-cache", create_if_missing=True)

image = (
    modal.Image.debian_slim()
    .pip_install("fastapi[standard]", "transformers>=4.37.0", "torch", "accelerate", "peft")
    .env({"HF_HUB_CACHE": "/cache"})
    .add_local_dir("schema", remote_path="/root/schema")
    .add_local_file("engine.py", remote_path="/root/engine.py")
    .add_local_file("service.py", remote_path="/root/service.py")
)

web_app = FastAPI(title="Qwen Session Orchestration Gateway")


@app.cls(
    image=image,
    gpu="L4",                      # 1. 💡 SWITCH TO L4: Cheaper hourly rate, great performance
    volumes={"/cache": hf_volume},
    enable_memory_snapshot=False,  # Keep this False so it initializes directly on the GPU
    min_containers=0,              # 2. 💸 CRITICAL FOR BUDGET: Scale to 0 when idle so you don't burn cash
    max_containers=2,              # 3. Prevent accidental cost spikes if traffic floods in
    container_idle_timeout=300     # 4. Keep GPU warm for 5 minutes after a hit for instant replies
)
class ModelServer:
    @modal.enter(snap=True)
    def setup_system(self):
        from engine import QwenInferenceEngine
        from service import SessionOrchestrator

        self.engine = QwenInferenceEngine(model_id="NYW619/outputs", cache_dir="/cache")
        self.engine.initialize()
        self.orchestrator = SessionOrchestrator(engine=self.engine)

    @modal.fastapi_endpoint(method="POST")
    def analyze_session(self, payload: dict):
        from schema.SessionRequest import SessionRequest

        validated_request = SessionRequest(**payload)
        business_response = self.orchestrator.process_chat_turn(validated_request)
        return business_response.model_dump()
