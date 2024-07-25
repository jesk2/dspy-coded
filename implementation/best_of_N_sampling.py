from judge import ListwiseRanking
import dspy

class BestofNSampling(dspy.Module):
    def __init__(self, model, rubric_template):
        super().__init__()
        self.judge = ListwiseRanking(model=model, rubric_template=rubric_template)
        
    def forward(self, instructions, response_list, reference_answers, rubric_data, num):
        all_rankings = self.judge.forward(instructions, response_list, reference_answers, rubric_data)
        top_n = []
        
        for i, rankings in enumerate(all_rankings):
            response_rankings = list(zip(response_list[i], rankings))
            sorted_responses = sorted(response_rankings, key=lambda x: x[1])
            top_responses = sorted_responses[:num] if num < len(sorted_responses) else sorted_responses
            top_n.append([response for response, rank in top_responses])

        return top_n

