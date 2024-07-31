from typing import List, Dict, Optional, Union
from pydantic import BaseModel
import dspy
from judge import DirectAssessment
from prometheus_eval.prompts import SCORE_RUBRIC_TEMPLATE


# Pydantic Models
class ResponseData(BaseModel):
    instruction: str
    response: str
    score: float


class InstructionData(BaseModel):
    instruction: str
    response: str
    score: float


class FilterCriteria(BaseModel):
    criteria: str
    score1_description: str
    score2_description: str
    score3_description: str
    score4_description: str
    score5_description: str


class DataFilter(dspy.Module):
    """
    Base class for data filtering operations.
    """

    def __init__(self) -> None:
        super().__init__()


class QualityResponseFilter(DataFilter):
    """
    Filters responses based on their quality scores.

    Attributes:
        judge: An instance of DirectAssessment used to evaluate responses.
        score_threshold: The minimum score required to consider a response as high quality.
        rubric_data: Data for generating the rubric.
    """

    def __init__(self, model, score_threshold: int) -> None:
        """
        Initialize the QualityResponseFilter class.

        Args:
            model: The model used for assessment.
            score_threshold: Minimum score required to consider a response as high quality.
        """
        super().__init__()
        self.judge = DirectAssessment(
            model=model, rubric_template=SCORE_RUBRIC_TEMPLATE
        )
        self.score_threshold = score_threshold
        self.rubric_data = FilterCriteria(
            criteria="Evaluate the quality of the response on these three metrics: clear and accurate reply, relevant and thorough information, and polite tone.",
            score1_description="The response is unclear in providing an accurate reply to the instruction, is incomplete, and has an impolite tone.",
            score2_description="The response provides a somewhat clear and accurate reply, but is incomplete, has irrelevant information, and/or is impolite.",
            score3_description="The response is helpful and accurate, somewhat detailed and relevant to the instruction, with a neutral or polite tone.",
            score4_description="The response is overall helpful, clear and accurate. It includes many relevant details in a thorough way, and uses a polite tone.",
            score5_description="The response is extremely clear and helpful, with very accurate information. It includes all relevant details in a thorough way, and uses a polite tone.",
        )

    def forward(
        self, instructions: List[str], responses: List[List[str]]
    ) -> List[ResponseData]:
        """
        Filter responses based on their scores.

        Args:
            instructions: List of instructions for each set of responses.
            responses: List of lists where each inner list contains responses for an instruction.

        Returns:
            A list of ResponseData containing instructions, responses, and their scores that meet the threshold.
        """
        _, all_scores = self.judge.forward(
            instructions, responses, self.rubric_data.model_dump(), [None] * len(instructions)
        )
        quality_responses = []
    
        for i, scores in enumerate(all_scores): 
            response_dict = [
                ResponseData(
                    instruction=instructions[i],
                    response=responses[i][j],
                    score=scores[j],
                )
                for j in range(len(scores))
                if scores[j] >= self.score_threshold
            ]
            quality_responses.extend(response_dict)

        return quality_responses


class DifficultyFilter(DataFilter):
    """
    Filters instructions based on their difficulty scores.

    Attributes:
        judge: An instance of DirectAssessment used to evaluate instructions.
        score_threshold: The minimum score required to consider an instruction as challenging.
        rubric_data: Data for generating the rubric.
    """

    def __init__(self, model, score_threshold: int) -> None:
        """
        Initialize the DifficultyFilter class.

        Args:
            model: The model used for assessment.
            score_threshold: Minimum score required to consider an instruction as challenging.
        """
        super().__init__()
        self.score_threshold = score_threshold
        self.judge = DirectAssessment(
            model=model, rubric_template=SCORE_RUBRIC_TEMPLATE
        )
        self.rubric_data = FilterCriteria(
            criteria="Evaluate the difficulty of the instruction on these three metrics: clearly written and specific language, definitively challenging instruction in the relevant field, and creative and relevant problem-solving.",
            score1_description="The instruction is unclear and too easy.",
            score2_description="The instruction is somewhat unclear but still easy.",
            score3_description="The instruction is moderately clear and lacks challenging problem-solving.",
            score4_description="The instruction is clear and adequately challenging, somewhat requiring creativity or problem-solving skills.",
            score5_description="The instruction is very clear and highly challenging, requiring much creativity and problem-solving skills.",
        )

    def forward(
        self, metaprompt: Union[str, List[str]], instructions: List[str]
    ) -> List[InstructionData]:
        """
        Filter instructions based on their difficulty scores.

        Args:
            metaprompt: A string or list of strings used as the metaprompt for the evaluation.
            instructions: List of instructions to be evaluated.

        Returns:
            A list of InstructionData containing instructions, responses, and their scores that meet the threshold.
        """
        responses = [[instr] for instr in instructions]
        _, all_scores = self.judge.forward(
            [metaprompt] * len(instructions),
            responses,
            self.rubric_data.model_dump(),
            [None] * len(instructions),
        )
        difficult_instructions = []

        # Ensure that the lengths match
        if len(all_scores) != len(instructions):
            raise ValueError("Scores and instructions length mismatch.")
    
        for i, scores in enumerate(all_scores):
            if len(scores) != 1:
                raise ValueError(f"Scores length mismatch for instruction {i}.")

            if scores[0] >= self.score_threshold:
                difficult_instructions.append(
                    InstructionData(
                        instruction=instructions[i],
                        response=instructions[i],
                        score=scores[0],
                    )
                )

        return difficult_instructions
