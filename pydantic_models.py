from pydantic import BaseModel
from typing import List, Optional

class Condition(BaseModel):
    condition: str
    rationale: str
    confidence: str
    relative_score: Optional[float] = 0.0

class SymptomResponse(BaseModel):
    input: str
    probable_conditions: List[Condition]
    recommended_next_steps: List[str]
    disclaimer: str
