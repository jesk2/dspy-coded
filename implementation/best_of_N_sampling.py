from typing import List, Dict
import dspy
from judge import DirectAssessment, ListwiseRanking


class BestofNSampling(dspy.Module):
    def __init__(self, model, rubric_template: str) -> None:
        """
        Initialize the BestofNSampling class.

        Args:
            model: The model used for assessment and ranking.
            rubric_template: The template for generating the rubric.
        """
        super().__init__()
        self.direct_assessment = DirectAssessment(
            model=model, rubric_template=rubric_template
        )
        self.listwise_ranking = ListwiseRanking(
            model=model, rubric_template=rubric_template
        )

    def forward(
        self,
        instructions: List[str],
        response_list: List[List[str]],
        rubric_data: Dict[str, str],
        reference_answers: List[str],
        num: int,
    ) -> List[List[str]]:
        """
        Perform Best-of-N sampling on the given responses based on their scores and ranking.

        Args:
            instructions: A list of instructions for each set of responses.
            response_list: A list of lists where each inner list contains responses for an instruction.
            rubric_data: A dictionary containing data for generating the rubric.
            reference_answers: A list of reference answers corresponding to each instruction.
            num: The number of top responses to select from each response list.

        Returns:
            A list of lists where each inner list contains the top N responses selected.
        """
        all_scores = []
        for responses in response_list:
            _, score_list = self.direct_assessment.forward(
                instructions, responses, rubric_data, reference_answers
            )
            all_scores.append(score_list)
        print(all_scores)

        top_n = []

        max_score = 5
        min_score = 1
        for response_sublist, score_list in zip(response_list, all_scores):
            # Create a dictionary to group responses by their scores
            score_buckets = {i: [] for i in range(min_score, max_score+1)}
            for response, score in zip(response_sublist, score_list):
                score_buckets[score].append(response)

            selected_responses = []

            for score in range(5, 0, -1):
                if score_buckets[score]:
                    responses_needed = num - len(selected_responses)
                    if responses_needed > 0:
                        selected_responses.extend(
                            score_buckets[score][:responses_needed]
                        )
                    if len(selected_responses) == num:
                        break

            # If more responses in bucket than num, rank them
            if len(selected_responses) > num:
                ranked_indices = self.listwise_ranking.forward(
                    [instructions[0]],  # Assuming single instruction for the ranking
                    [selected_responses],
                    rubric_data,
                    [
                        reference_answers[0]
                    ],  # Assuming single reference answer for the ranking
                )[0]
                selected_responses = [
                    selected_responses[j]
                    for j in sorted(
                        range(len(selected_responses)), key=lambda x: ranked_indices[x]
                    )[:num]
                ]

            top_n.append(selected_responses)

        return top_n
