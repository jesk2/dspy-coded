from judge import ListwiseRanking
import dspy
import random

class BestofNSampling(dspy.Module):
    def __init__(self, model, rubric_template):
        super().__init__()
        self.judge = ListwiseRanking(model=model, rubric_template=rubric_template)
        
    def forward(self, instructions, response_list, reference_answers, rubric_data, num):
        all_rankings = self.judge.forward(instructions, response_list, reference_answers, rubric_data)
        best_n_responses = []
        
        for i, rankings in enumerate(all_rankings):
            responses = response_list[i]

            # Map rank to responses
            rank_to_responses = {}
            for response, rank in zip(responses, rankings):
                if rank not in rank_to_responses:
                    rank_to_responses[rank] = []
                rank_to_responses[rank].append(response)

            # Sort ranks in ascending order
            sorted_ranks = sorted(rank_to_responses.keys())

            selected_responses = []
            remaining_n = num
            for rank in sorted_ranks:
                responses_at_rank = rank_to_responses[rank]
                if remaining_n > len(responses_at_rank):
                    selected_responses.extend(responses_at_rank)
                    remaining_n -= len(responses_at_rank)
                else:
                    selected_responses.extend(random.sample(responses_at_rank, remaining_n))
                    remaining_n = 0
                    break

            best_n_responses.append(selected_responses)

        return best_n_responses