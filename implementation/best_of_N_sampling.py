from judge import DirectAssessment, ListwiseRanking
import dspy

class BestofNSampling(dspy.Module):
    def __init__(self, model, rubric_template):
        super().__init__()
        self.direct_assessment = DirectAssessment(model=model, rubric_template=rubric_template)
        self.listwise_ranking = ListwiseRanking(model=model, rubric_template=rubric_template)
        
    def forward(self, instructions, response_list, rubric_data, reference_answers, num):
        all_scores = []
        for responses in response_list: 
            _, score_list = self.direct_assessment.forward(instructions, responses, rubric_data, reference_answers)
            all_scores.append(score_list)
        print(all_scores)

        top_n = []

        for response_sublist, score_list in zip(response_list, all_scores):
            # Create a dictionary to group responses by their scores
            score_buckets = {num: [] for num in range(1, 6)}
            for response, score in zip(response_sublist, score_list):
                score_buckets[score].append(response)
        
            selected_responses = []

            for score in range(5, 0, -1):
                if score_buckets[score]:
                    responses_needed = num - len(selected_responses)
                    if responses_needed > 0:
                        selected_responses.extend(score_buckets[score][:responses_needed])
                    if len(selected_responses) == num:
                        break
            
            # more responses in bucket than num 
            if len(selected_responses) > num:
                ranked_indices = self.listwise_ranking.forward(
                    [instructions[i]], 
                    [selected_responses], 
                    rubric_data, 
                    [reference_answers[i]]
                )[0]
                selected_responses = [selected_responses[j] for j in sorted(range(len(selected_responses)), key=lambda x: ranked_indices[x])[:num]]
            
            top_n.append(selected_responses)

        return top_n
    
