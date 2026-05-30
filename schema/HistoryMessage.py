from pydantic import BaseModel, Field


class HistoryMessage(BaseModel):
    role: str = Field(..., description="The speaker's role, e.g., 'user' or 'assistant'.")
    message: str = Field(..., description="The actual text content of the message.")
