from engine import QwenInferenceEngine
from schema.BusinessResponse import BusinessResponse
from schema.SessionRequest import SessionRequest


class SessionOrchestrator:
    def __init__(self, engine: QwenInferenceEngine):
        self.engine = engine

    def process_chat_turn(self, payload: SessionRequest) -> BusinessResponse:
        messages = [{"role": "system", "content": f"The user's current goal is: '{payload.current_goal}'. Intent "
                                                  f"attempt: {payload.attemptCount}."}]
        for chat in payload.conversation_history:
            messages.append({"role": chat.role, "content": chat.message})

        raw_text_prompt = self.engine.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        raw_output = self.engine.generate_text(raw_text_prompt)

        return BusinessResponse(
            assistant_message=raw_output.strip(),
            goal_completed=payload.goal_completed,
            extracted_data={"sessionID_context": payload.sessionID},
            next_goal=payload.current_goal
        )