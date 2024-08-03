import dspy
from pydantic import BaseModel
from typing import List, Tuple, Dict

class InstructionData(BaseModel):
    instruction: str
    responses: List[str]
    reference_answer: str


class LLMsAsJudge(dspy.Module):
    def __init__(self, model, rubric_template: str):
        """
        Initialize the LLMsAsJudge class.

        Args:
            model: The language model used for evaluation.
            rubric_template: The template for generating the rubric.
        """
        super().__init__()
        self.model = model
        self.rubric_template = rubric_template


class DirectAssessment(LLMsAsJudge):
    def forward(
        self,
        instructions: List[str],
        responses: List[str],
        rubric_data: Dict[str, str],
        reference_answers: List[str],
    ) -> Tuple[List[str], List[float]]:
        """
        Perform direct assessment on the responses.

        Args:
            instructions: A list of instructions.
            responses: A list of lists where each inner list contains responses for an instruction.
            rubric_data: A dictionary containing data for generating the rubric.
            reference_answers: A list of reference answers corresponding to each instruction.

        Returns:
            A tuple containing a list of feedbacks and a list of scores.
        """
        rubric = self.rubric_template.format(**rubric_data)

        feedbacks, scores = self.model.absolute_grade(
            instructions=instructions,
            responses=responses,
            rubric=rubric,
            reference_answers=reference_answers,
        )

        return feedbacks, scores


class PairwiseRanking(LLMsAsJudge):
    def forward(
        self,
        instructions: List[str],
        responseA: List[str],
        responseB: List[str],
        rubric_data: Dict[str, str],
        reference_answers: List[str],
    ) -> Tuple[List[str], List[str]]:
        """
        Perform pairwise ranking of responses.

        Args:
            instructions: A list of instructions.
            responseA: A list of responses from set A.
            responseB: A list of responses from set B.
            rubric_data: A dictionary containing data for generating the rubric.
            reference_answers: A list of reference answers corresponding to each instruction.

        Returns:
            A tuple containing a list of feedbacks and a list of winners.
        """
        rubric = self.rubric_template.format(**rubric_data)

        feedbacks, winners = self.model.relative_grade(
            instructions=instructions,
            responses_A=responseA,
            responses_B=responseB,
            rubric=rubric,
            reference_answers=reference_answers,
        )

        return feedbacks, winners


class ListwiseRanking(PairwiseRanking):
    def forward(
        self,
        instructions: List[str],
        response_list: List[List[str]],
        rubric_data: Dict[str, str],
        reference_answers: List[str],
    ) -> List[List[int]]:
        """
        Perform listwise ranking of responses.

        Args:
            instructions: A list of instructions.
            response_list: A list of lists where each inner list contains responses for an instruction.
            rubric_data: A dictionary containing data for generating the rubric.
            reference_answers: A list of reference answers corresponding to each instruction.

        Returns:
            A list of lists where each inner list contains the rankings of responses.
        """
        all_rankings = []

        for i, responses in enumerate(response_list):
            if len(responses) < 2:
                all_rankings.append([1] * len(responses))
                continue

            win_counts = {response: 0 for response in responses}
            instruction_responses_pairs = [
                (
                    [instructions[i]] * 2,
                    [responses[j], responses[k]],
                    [reference_answers[i]],
                )
                for j in range(len(responses))
                for k in range(j + 1, len(responses))
            ]

            # Flatten the lists for batch processing
            flat_instructions = [pair[0] for pair in instruction_responses_pairs]
            flat_responsesA = [pair[1][0] for pair in instruction_responses_pairs]
            flat_responsesB = [pair[1][1] for pair in instruction_responses_pairs]
            flat_reference_answers = [pair[2] for pair in instruction_responses_pairs]

            _, winners = super().forward(
                flat_instructions,
                flat_responsesA,
                flat_responsesB,
                rubric_data,
                flat_reference_answers,
            )

            idx = 0
            for j in range(len(responses)):
                for k in range(j + 1, len(responses)):
                    responseA = responses[j]
                    responseB = responses[k]

                    if winners[idx][0] == "A":
                        win_counts[responseA] += 1
                    else:
                        win_counts[responseB] += 1
                    idx += 1

            rankings = [0] * len(responses)
            sorted_responses = sorted(
                win_counts.items(), key=lambda item: item[1], reverse=True
            )

            rank = 1
            for idx in range(len(sorted_responses)):
                if idx > 0 and sorted_responses[idx][1] < sorted_responses[idx - 1][1]:
                    rank = idx + 1
                rankings[responses.index(sorted_responses[idx][0])] = rank

            all_rankings.append(rankings)

        return all_rankings
