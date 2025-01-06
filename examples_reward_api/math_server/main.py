import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from .utils_math import compute_score
from loguru import logger

app = FastAPI()

class RewardRequest(BaseModel):
    solutions: List[str]
    ground_truths: Optional[List[str]] = None
    metadata: Optional[Dict[str, Any]] = None
    
class RewardResponse(BaseModel):
    scores: List[float]
        

@app.post("/reward")
def compute_reward(item: RewardRequest) -> RewardResponse:
    solutions = item.solutions
    ground_truths = item.ground_truths
    metadata = item.metadata
    
    scores = [
        compute_score(solution, ground_truth) for solution, ground_truth in zip(solutions, ground_truths)
        ]
    
    logger.info(f"Computed scores: {scores}")
    
    return RewardResponse(scores=scores)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)