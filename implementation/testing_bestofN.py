from prometheus_eval.prompts import SCORE_RUBRIC_TEMPLATE, RELATIVE_PROMPT
from judge import ListwiseRanking
from best_of_N_sampling import BestofNSampling
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

rubric_template = "{criteria}\n1: {score1_description}\n2: {score2_description}\n3: {score3_description}\n4: {score4_description}\n5: {score5_description}"

# Initialize the mock model and BestofNSampling class
mock_model = MockModel()
listwise_ranking = ListwiseRanking(model=mock_model, rubric_template=SCORE_RUBRIC_TEMPLATE)
best_of_n_sampling = BestofNSampling(model=mock_model, rubric_template=rubric_template)
best_of_n_sampling.judge = listwise_ranking

# Test BestofNSampling
num_responses = 3  # Number of top responses to return
top_responses = best_of_n_sampling.forward(instructions, responses, reference_answers, rubric_data, num_responses)
for i, top in enumerate(top_responses):
    print(f"Instruction: {instructions[i]}")
    print(f"Top {num_responses} Responses: {top}")
    print()