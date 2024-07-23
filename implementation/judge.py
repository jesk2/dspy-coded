import dspy
from prometheus_eval.prompts import ABSOLUTE_PROMPT, SCORE_RUBRIC_TEMPLATE, RELATIVE_PROMPT

class LLMsAsJudge(dspy.Module):
    def __init__(self, model, rubric_template):
        super().__init__()
        self.model = model
        self.rubric_template = rubric_template

class DirectAssessment(LLMsAsJudge):
    def forward(self, instructions, responses, rubric_data, reference_answers):
        rubric = self.rubric_template.format(**rubric_data)
        all_feedbacks = []
        all_scores = []

        for instruction, response_list, reference_answer in zip(instructions, responses, reference_answers):
            for response in response_list:
                feedbacks, score = self.model.absolute_grade(
                    instructions=[instruction],
                    responses=[response],
                    rubric=rubric,
                    reference_answers=[reference_answer]
                )
                all_feedbacks.extend(feedbacks)
                all_scores.extend(score)
        
        return all_feedbacks, all_scores

class PairwiseRanking(LLMsAsJudge):
    def forward(self, instructions, responseA, responseB, rubric_data, reference_answers):
        rubric = self.rubric_template.format(**rubric_data)
        feedbacks, winners = self.model.relative_grade(
            instructions=instructions,
            responses_A=responseA,
            responses_B=responseB,
            rubric=rubric,
            reference_answers=reference_answers
        )

        return feedbacks, winners

class ListwiseRanking(PairwiseRanking):
    def forward(self, instructions, response_list, rubric_data, reference_answers):
        rubric = self.rubric_template.format(**rubric_data)
        all_sorted_responses = []
        for i, responses in enumerate(response_list):
            if len(responses) < 2:
                continue

            win_counts = {response: 0 for response in responses}
            # O(n^2)
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
            all_sorted_responses.append(sorted_responses)

        # returns all n repsonses 
        return all_sorted_responses
