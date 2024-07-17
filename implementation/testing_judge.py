from prometheus_eval.prompts import SCORE_RUBRIC_TEMPLATE, RELATIVE_PROMPT
from judge import PairwiseRanking, ListwiseRanking
import random
from typing import Any, Dict, List, Tuple, Union

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
    ) -> Tuple[List[str], List[int]]:
        feedbacks = ["Good", "Average"]  # Example feedbacks
        scores = [random.randint(1, 5) for _ in range(2)] 
        return feedbacks, scores

# Define the rubric data
rubric_data = {
    "criteria": "Is the response accurate and relevant to the question?",
    "score1_description": "The response is completely irrelevant or incorrect.",
    "score2_description": "The response is mostly irrelevant or incorrect.",
    "score3_description": "The response is somewhat relevant but contains inaccuracies.",
    "score4_description": "The response is mostly relevant and mostly accurate.",
    "score5_description": "The response is completely relevant and accurate."
}


# Prepare test data
instructions = [
    "What's the capital of France?",
    "How many continents are there?",
    "Name a programming language.",
    "What is the largest planet in our solar system?"
]

responses = [
    ["Paris", "London", "Berlin", "Madrid", "Seoul", "Tokyo", "Singapore"],
    ["Seven", "Eight", "Six"],
    ["Python", "Java", "C++", "Ruby", "JavaScript"],
    ["Jupiter", "Saturn", "Earth", "Mars", "Venus", "Neptune"]
]

reference_answers = [
    "Paris",
    "Seven",
    "Python",
    "Jupiter"
]

# Initialize the mock model and PairwiseRanking and ListwiseRanking classes
mock_model = MockModel()
pairwise_ranking = PairwiseRanking(model=mock_model, rubric_template=SCORE_RUBRIC_TEMPLATE)
listwise_ranking = ListwiseRanking(model=mock_model, rubric_template=SCORE_RUBRIC_TEMPLATE)

# Test PairwiseRanking
for i in range(len(instructions)):
    feedback, score = pairwise_ranking.forward(instructions[i], responses[i][0], responses[i][1], reference_answers[i], rubric_data)
    print(feedback, score)
    print(f"Instruction: {instructions[i]}")
    print(f"Response A: {responses[i][2]}")
    print(f"Response B: {responses[i][0]}")
    print(f"Winner: {'A' if score == 'A' else 'B'}")
    print()

# Test ListwiseRanking
best_responses = listwise_ranking.forward(instructions, responses, reference_answers, rubric_data, num_responses=2)
for i, best in enumerate(best_responses):
    print(f"Instruction: {instructions[i]}")
    print(f"Best Response: {best}")
    print()