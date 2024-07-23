from prometheus_eval.prompts import SCORE_RUBRIC_TEMPLATE, RELATIVE_PROMPT
from judge import PairwiseRanking, ListwiseRanking
from mock import MockModel

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

responses = [ # for listwise rnaking 
    ["Paris", "London", "Berlin", "Madrid", "Seoul", "Tokyo", "Singapore"],
    ["Seven", "Eight", "Six"],
    ["Python", "Java", "C++", "Ruby", "JavaScript"],
    ["Jupiter", "Saturn", "Earth", "Mars", "Venus", "Neptune"]
]

responses_A = [
    "Paris", "Seven", "Python", "Jupiter"
]

responses_B = [
    "London", "Eight", "Java", "Saturn"
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
feedbacks, winners = pairwise_ranking.forward(instructions, responses_A, responses_B, reference_answers, rubric_data)
print(f"All winners: {winners}")
for i, winner in enumerate(winners):
    print(f"Instruction: {instructions[i]}")
    print(f"Response A: {responses_A[i]}")
    print(f"Response B: {responses_B[i]}")
    print(f"Pairwise Winners: {winner}")
    print()

# Test ListwiseRanking
best_responses = listwise_ranking.forward(instructions, responses, reference_answers, rubric_data)
for i, best in enumerate(best_responses):
    print(f"Instruction: {instructions[i]}")
    print(f"Listwise Best Response: {best}")
    print()