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
        # logging.debug(f'DirectAssessment forward: instruction={instruction}, response={response}, score={score}')
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
        # returns one winner 
        if scores[0] > scores[1]:
            winner = 'A'
        else:
            winner = 'B'
        return feedbacks, winner

class ListwiseRanking(PairwiseRanking):
    def forward(self, instructions, response_list, reference_answers, rubric_data, num_responses=1):
        best_responses = []
        for i, responses in enumerate(response_list):
            if len(responses) < 2:
                # logging.warning(f"Not enough responses for instruction {i + 1}. Skipping.")
                continue

            while len(responses) > 1 and len(responses) > num_responses:
                next_round = []
                for j in range(0, len(responses) - 1, 2):
                    responseA = responses[j]
                    responseB = responses[j + 1]
                    _, winner = super().forward(instructions[i], responseA, responseB, reference_answers[i], rubric_data)
                    if winner == 'A':
                        next_round.append(responseA)
                    else:
                        next_round.append(responseB)
                if len(responses) % 2 == 1:  # If odd number of responses, carry the last one to next round
                    next_round.append(responses[-1])
                responses = next_round

            if responses:
                best_responses.append(responses[:num_responses])

        return best_responses
