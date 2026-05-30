from pydantic import BaseModel, Field
from typing import List
from schema.HistoryMessage import HistoryMessage


class SessionRequest(BaseModel):
    sessionID: str = Field(..., description="Unique customer tracking sequence ID.")
    current_goal: str = Field(..., description="The objective the user is currently attempting to complete.")
    conversation_history: List[HistoryMessage] = Field(default=[], description="The historical message stack.")
    goal_completed: bool = Field(default=False, description="Flag indicating if the goal state is satisfied.")
    attemptCount: int = Field(default=1, description="Number of turns taken to satisfy the current goal.")
