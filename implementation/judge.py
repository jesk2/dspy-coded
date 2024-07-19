import dspy
from prometheus_eval.prompts import ABSOLUTE_PROMPT, SCORE_RUBRIC_TEMPLATE, RELATIVE_PROMPT
import logging 

logging.basicConfig(level=logging.DEBUG)

class LLMsAsJudge(dspy.Module):
    def __init__(self, model, rubric_template):
        super().__init__()
        self.model = model
        self.rubric_template = rubric_template

class DirectAssessment(LLMsAsJudge):
    def forward(self, instructions, responses, reference_answers, rubric_data):
        rubric = self.rubric_template.format(**rubric_data)
        feedbacks, score = self.model.absolute_grade(
            instructions=instructions,
            responses=responses,
            rubric=rubric,
            reference_answers=reference_answers
        )
        return feedbacks, score 

class PairwiseRanking(LLMsAsJudge):
    def forward(self, instructions, responseA, responseB, reference_answer, rubric_data):
        rubric = self.rubric_template.format(**rubric_data)
        feedbacks, scores = self.model.relative_grade(
            instructions=[instructions],
            responses_A=[responseA],
            responses_B=[responseB],
            rubric=rubric_data,
            reference_answers=[reference_answer]
        )
        if scores[0] > scores[1]:
            winner = 'A'
        else:
            winner = 'B'
        return feedbacks, winner

class ListwiseRanking(PairwiseRanking):
    def forward(self, instructions, response_list, reference_answers, rubric_data, num_responses):
        best_responses = []
        for i, responses in enumerate(response_list):
            if len(responses) < 2:
                continue

            win_counts = {response: 0 for response in responses}

            for j in range(len(responses)):
                for k in range(j + 1, len(responses)):
                    responseA = responses[j]
                    responseB = responses[k]
                    _, winner = super().forward(instructions[i], responseA, responseB, reference_answers[i], rubric_data)
                    if winner == 'A':
                        win_counts[responseA] += 1
                    else:
                        win_counts[responseB] += 1

            sorted_responses = sorted(win_counts.keys(), key=lambda x: win_counts[x], reverse=True)
            best_responses.append(sorted_responses[:num_responses])

        return best_responses
