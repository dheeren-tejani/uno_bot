from typing import Literal
from pydantic import BaseModel, Field


class StartRequest(BaseModel):
    difficulty: Literal["easy", "normal", "hard"]


class ActionRequest(BaseModel):
    game_id: str = Field(min_length=8, max_length=64)
    action: int = Field(ge=0, le=61)