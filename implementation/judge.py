from typing import List, Dict, Optional, Tuple, Any
from pydantic import BaseModel
import dspy

class PointwiseData(BaseModel):
    """
    Model representing pointwise data used for assessments.
    
    Attributes:
        instruction (str): The instruction for the assessment.
        response (str): The response to be assessed.
        reference_answer (Optional[str]): The reference answer for comparison, if available.
    """
    instruction: str
    response: str
    reference_answer: Optional[str] = None

class PairwiseData(BaseModel):
    """
    Model representing pairwise data used for assessments.

    Attributes:
        instruction (str): The instruction for the assessment.
        responseA (str): The first response to be compared.
        responseB (str): The second response to be compared.
        reference_answer (Optional[str]): The reference answer for comparison, if available.
    """
    instruction: str
    responseA: str
    responseB: str
    reference_answer: Optional[str] = None

class LLMsAsJudge(dspy.Module):
    """
    Base class for modules that utilize language models for judging responses.

    Attributes:
        model (Any): The model used for grading or ranking responses.
        rubric_template (str): Template for the rubric used in assessments.
    """
    def __init__(self, model: Any, rubric_template: str):
        super().__init__()
        self.model = model
        self.rubric_template = rubric_template

class DirectAssessment(LLMsAsJudge):
    """
    Assessment module for absolute grading.

    Inherits from LLMsAsJudge and implements forward method for grading responses.
    """
    def forward(self, instructions: List[str], responses: List[str], rubric_data: Dict[str, Any], reference_answers: List[Optional[str]]) -> Tuple[List[str], List[float]]:
        """
        Grades responses based on the provided instructions and rubric.

        Args:
            instructions (List[str]): List of instructions for the assessment.
            responses (List[str]): List of responses to be assessed.
            rubric_data (Dict[str, Any]): Data to populate the rubric template.
            reference_answers (List[Optional[str]]): List of reference answers for comparison.

        Returns:
            Tuple[List[str], List[float]]: Feedbacks and scores for the responses.
        """
        rubric = self.rubric_template.format(**rubric_data)
        all_feedbacks = []
        all_scores = []

        for instruction, response, reference_answer in zip(instructions, responses, reference_answers):
            feedbacks, scores = self.model.absolute_grade(
                instructions=[instruction],
                responses=[response],
                rubric=rubric,
                reference_answers=[reference_answer]
            )
            all_feedbacks.extend(feedbacks)
            all_scores.extend(scores)

        return all_feedbacks, all_scores    

class PairwiseRanking(LLMsAsJudge):
    """
    Assessment module for pairwise ranking.

    Inherits from LLMsAsJudge and implements forward method for comparing two responses.
    """
    def forward(self, instructions: List[str], responseA: List[str], responseB: List[str], rubric_data: Dict[str, Any], reference_answers: List[Optional[str]]) -> Tuple[List[str], List[str]]:
        """
        Compares two sets of responses and determines the better one.

        Args:
            instructions (List[str]): List of instructions for the assessment.
            responseA (List[str]): List of first responses for pairwise comparison.
            responseB (List[str]): List of second responses for pairwise comparison.
            rubric_data (Dict[str, Any]): Data to populate the rubric template.
            reference_answers (List[Optional[str]]): List of reference answers for comparison.

        Returns:
            Tuple[List[str], List[str]]: Feedbacks and winners for each pair of responses.
        """
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
    """
    Assessment module for listwise ranking based on pairwise comparisons.

    Inherits from PairwiseRanking and implements forward method for ranking a list of responses.
    """
    def forward(self, instructions: List[str], response_list: List[List[str]], rubric_data: Dict[str, Any], reference_answers: List[Optional[str]]) -> List[List[int]]:
        """
        Ranks a list of responses based on pairwise comparisons.

        Args:
            instructions (List[str]): List of instructions for the assessment.
            response_list (List[List[str]]): List of response lists to be ranked.
            rubric_data (Dict[str, Any]): Data to populate the rubric template.
            reference_answers (List[Optional[str]]): List of reference answers for comparison.

        Returns:
            List[List[int]]: Rankings for each list of responses.
        """
        all_rankings = []

        for i, responses in enumerate(response_list):
            if len(responses) < 2:
                all_rankings.append([1] * len(responses))
                continue

            win_counts = {response: 0 for response in responses}
            for j in range(len(responses)):
                for k in range(j + 1, len(responses)):
                    responseA = responses[j]
                    responseB = responses[k]

                    _, winners = super().forward([instructions[i]], [responseA], [responseB], rubric_data, reference_answers[i])

                    if winners[0] == 'A':
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
