import random
from typing import Any, Dict, List, Tuple

class MockModel:
    def relative_grade(
        self,
        *,
        instructions: List[str],
        responses_A: List[str],
        responses_B: List[str],
        rubric: List[str] | str,
        reference_answers: List[str] = None,
        params: Dict[str, Any] = {},
    ) -> Tuple[List[str], List[List[int]]]:
        feedbacks = ["Good", "Average"] * len(instructions)  # Example feedbacks
        scores = [[random.randint(1, 5), random.randint(1, 5)] for _ in range(len(instructions))]
        return feedbacks, scores
