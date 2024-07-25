from prometheus_eval.vllm import VLLM
from prometheus_eval import PrometheusEval
from prometheus_eval.prompts import SCORE_RUBRIC_TEMPLATE, RELATIVE_PROMPT
from judge import DirectAssessment, PairwiseRanking, ListwiseRanking


# Initialize the model
model = VLLM('prometheus-eval/prometheus-7b-v2.0', max_model_len=4096, tensor_parallel_size=1, gpu_memory_utilization=0.6, download_dir="/mnt/sda/jessica/cache")
judge = PrometheusEval(model=model, relative_grade_template=RELATIVE_PROMPT)

# Define the rubric data
rubric_data = {
    "criteria": "Is the response accurate and relevant to the question?",
    "score1_description": "The response is completely irrelevant or incorrect.",
    "score2_description": "The response is mostly irrelevant or incorrect.",
    "score3_description": "The response is somewhat relevant but contains inaccuracies.",
    "score4_description": "The response is mostly relevant and mostly accurate.",
    "score5_description": "The response is completely relevant and accurate."
}

score_rubric = SCORE_RUBRIC_TEMPLATE.format(**rubric_data)

# Prepare test data
instructions = [
    "What's the capital of France?",
    "How many continents are there?",
    "Name a programming language.",
    "What is the largest planet in our solar system?"
]

responses = [ # for listwise ranking 
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

# Initialize PairwiseRanking and ListwiseRanking with the real model
direct_assessment = DirectAssessment(model=judge, rubric_template=SCORE_RUBRIC_TEMPLATE)
pairwise_ranking = PairwiseRanking(model=judge, rubric_template=SCORE_RUBRIC_TEMPLATE)
listwise_ranking = ListwiseRanking(model=judge, rubric_template=SCORE_RUBRIC_TEMPLATE)

# Test Direct Assessment 
feedback, scores = direct_assessment.forward(instructions, responses, rubric_data, reference_answers)
print("Direct Assessment Results:")
for i, instruction in enumerate(instructions):
    num_responses = len(responses[i])
    print(f"Instruction: {instruction}")
    for j in range(num_responses):
        print(f"Response: {responses[i][j]}")
        print(f"Feedback: {feedback[i][j]}")
        print(f"Score: {scores[i][j]}")
        print()

# Test PairwiseRanking
feedbacks, winners = pairwise_ranking.forward(instructions, responses_A, responses_B, rubric_data, reference_answers)
print("Pairwise Ranking Results:")
print(feedbacks)
for i, winner in enumerate(winners):
    print(f"Instruction: {instructions[i]}")
    print(f"Response A: {responses_A[i]}")
    print(f"Response B: {responses_B[i]}")
    print(f"Pairwise Winner: {winner}")
    print()

# Test ListwiseRanking
all_ranks = listwise_ranking.forward(instructions, responses, rubric_data, reference_answers)
print("Listwise Ranking Results:")
print(all_ranks)
for i, ranks in enumerate(all_ranks):
    print(f"Instruction: {instructions[i]}")
    print(f"Listwise Best Response: {ranks}")
    print()
