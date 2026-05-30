import modal
from fastapi import FastAPI

app = modal.App(name="qwen-session-api")
hf_volume = modal.Volume.from_name("qwen-model-cache", create_if_missing=True)

image = (
    modal.Image.debian_slim()
    .pip_install("fastapi[standard]", "transformers>=4.37.0", "torch", "accelerate", "peft")
    .add_local_dir("schema", remote_path="/root/schema")
    .add_local_file("engine.py", remote_path="/root/engine.py")
    .add_local_file("service.py", remote_path="/root/service.py")
    .env({"HF_HUB_CACHE": "/cache"})
)

web_app = FastAPI(title="Qwen Session Orchestration Gateway")


@app.cls(image=image, gpu="A10G", volumes={"/cache": hf_volume}, enable_memory_snapshot=True)
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
