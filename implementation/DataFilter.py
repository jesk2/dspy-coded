from judge import DirectAssessment, PairwiseRanking, ListwiseRanking
import dspy
from dspy import Prediction
import logging 

class DataFilter(dspy.Module):
    def __init__(self, direct_assessment):
        super().__init__()
        self.direct_assessment = direct_assessment

class ResponseFilter(DataFilter):
    def forward(self, instructions, responses, reference_answers, rubric_data):
        scored_responses = []
        for instruction, response, reference_answer in zip(instructions, responses, reference_answers):
            feedback, score = self.direct_assessment.forward(instruction, response, reference_answer, rubric_data)
            if score is None:
                logging.error(f'Score is None for instruction={instruction}, response={response}')
            scored_responses.append((response, score))
        
        # Filtering quality based on score: example 3 (can also implement as parameter?)
        high_quality_responses = [response for response, score in scored_responses if score is not None and score > 3]
        
        return high_quality_responses


class DifficultyFilter(DataFilter):
    def forward(self, instructions, reference_answers, rubric_data):
        scored_instructions = []
        for instruction, reference_answer in zip(instructions, reference_answers):
            feedback, score = self.direct_assessment.forward(instruction, "", reference_answer, rubric_data)
            if score is None:
                logging.error(f'Score is None for instruction={instruction}')
            scored_instructions.append((instruction, score))
        
        # Filtering quality based on score: example 3 (can also implement as parameter?)
        challenging_instructions = [instr for instr, score in scored_instructions if score is not None and score > 3]
        
        return challenging_instructions

class DiversityFilter(DataFilter):
    def forward(self, instructions, rubric_data):
        scored_instructions = []
        for instruction in instructions:
            feedback, score = self.direct_assessment.forward(instruction, "", "", rubric_data)
            if score is None:
                logging.error(f'Score is None for instruction={instruction}')
            scored_instructions.append((instruction, feedback))
        
        return scored_instructions
