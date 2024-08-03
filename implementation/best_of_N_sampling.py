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
        flat_instructions = []
        flat_responses = []
        for instr, responses in zip(instructions, response_list):
            flat_instructions.extend([instr] * len(responses))
            flat_responses.extend([[response] for response in responses])

        # Obtain scores from direct assessment
        _, all_scores = self.direct_assessment.forward(
            flat_instructions, flat_responses, rubric_data, [None] * len(flat_instructions)
        )

        # Split the scores back into the original structure
        split_scores = []
        idx = 0
        for responses in response_list:
            split_scores.append(all_scores[idx:idx + len(responses)])
            idx += len(responses)

        def process_responses(instr, response_sublist, score_list, ref_ans):
            score_buckets = {i: [] for i in range(1, 6)}  # Assuming scores are between 1 and 5
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

            if len(selected_responses) > num:
                ranked_indices = self.listwise_ranking.forward(
                    [instr],
                    [selected_responses],
                    rubric_data,
                    [ref_ans] if ref_ans is not None else [None]
                )[0]
                selected_responses = [
                    selected_responses[j]
                    for j in sorted(
                        range(len(selected_responses)), key=lambda x: ranked_indices[x]
                    )[:num]
                ]

            return selected_responses

        top_n = []
        if reference_answers is None:
            for instr, response_sublist, score_list in zip(instructions, response_list, split_scores):
                top_n.append(process_responses(instr, response_sublist, score_list, None))
        else:
            for instr, response_sublist, score_list, ref_ans in zip(instructions, response_list, split_scores, reference_answers):
                top_n.append(process_responses(instr, response_sublist, score_list, ref_ans))

        return top_n