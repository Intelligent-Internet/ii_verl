# Reward Model API Example

This is an example of how to use the reward API in VerL.
The reward API evaluates mathematical solutions (MATH dataset) against ground truth answers and returns scores (0 or 1) based on solution correctness.
The compute_score function is adapted from https://github.com/volcengine/verl/blob/main/verl/utils/reward_score/math.py

## Installation
```bash
pip install fastapi uvicorn pydantic loguru
```

## Usage

### Starting the Server

Run the server using:

```bash
cd examples_reward_api
python -m math_server.main
```

The server will start on `http://0.0.0.0:8000`

### API Endpoints

#### POST /reward

Computes reward scores for mathematical solutions.

**Request Body:**

```json
{
    "solutions": ["\\boxed{\\frac{1}{2}}", "\\boxed{2}"],
    "ground_truths": ["\\frac{1}{2}", "2"],
    "metadata": {} // Optional
}
```

**Response:**

```json
{
    "scores": [1.0, 1.0]
}
```

### Example

```python
import requests

data = {
    "solutions": ["\\boxed{\\frac{1}{2}}"],
    "ground_truths": ["\\frac{1}{2}"],
}

response = requests.post("http://localhost:8000/reward", json=data)
print(response.json())  # {"scores": [1.0]}
```