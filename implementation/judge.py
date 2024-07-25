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
    def forward(self, instructions, responses, rubric_data, reference_answers):
        rubric = self.rubric_template.format(**rubric_data)
        all_feedbacks = []
        all_scores = []

        for instruction, response_list, reference_answer in zip(instructions, responses, reference_answers):
            feedbacks_for_instruction = []
            scores_for_instruction = []
            for response in response_list:
                feedbacks, score = self.model.absolute_grade(
                    instructions=[instruction],
                    responses=[response],
                    rubric=rubric,
                    reference_answers=[reference_answer]
                )
                feedbacks_for_instruction.extend(feedbacks)
                scores_for_instruction.extend(score)
            all_feedbacks.append(feedbacks_for_instruction)
            all_scores.append(scores_for_instruction)
    
        return all_feedbacks, all_scores

class PairwiseRanking(LLMsAsJudge): # should only be single pair per instruction 
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
        all_rankings = []

        for i, responses in enumerate(response_list): # first response list 
            if len(responses) < 2:
                all_rankings.append([1] * len(responses))
                continue

            win_counts = {response: 0 for response in responses} # dict with win counts for each response after pairwise
            for j in range(len(responses)):
                for k in range(j+1, len(responses)):
                    responseA = responses[j]
                    responseB = responses[k]

                    _, winner = super().forward([instructions[i]], [responseA], [responseB], rubric_data, reference_answers[i])
                    
                    if winner[0] == 'A':
                        win_counts[responseA] += 1
                    else:
                        win_counts[responseB] += 1

            rankings = [0] * len(responses)
            sorted_responses = sorted(win_counts.items(), key=lambda item: item[1], reverse=True)

            rank = 1
            for idx in range(len(sorted_responses)):
                if idx > 0 and sorted_responses[idx][1] < sorted_responses[idx - 1][1]:
                    rank = idx + 1
                rankings[responses.index(sorted_responses[idx][0])] = rank

            all_rankings.append(rankings)

        return all_rankings
