"""
    Client function to compute reward scores for a batch of data by calling a remote API server.
    
    Expected API Server Design:
    - Endpoint: POST {api_url}. Example: http://localhost:8000/reward
    - Input JSON Format:
        {
            "solutions": [str, ...],      # List of solution to evaluate
            "ground_truths": [str, ...]   # List of corresponding ground truth
        }
    - Expected Response:
        {
            "scores": [float, ...]        # List of reward scores between 0.0 and 1.0
        }
"""

import requests
from typing import List

def compute_batch_scores(lst_solution_strs: List[str], lst_ground_truths: List[str], api_url: str) -> List[float]:
    payload = {
        "solutions": lst_solution_strs,
        "ground_truths": lst_ground_truths,
    }
    
    response = requests.post(api_url, json=payload)
    response.raise_for_status()
    
    return response.json()['scores']