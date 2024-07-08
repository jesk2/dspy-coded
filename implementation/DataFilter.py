from judge import DirectAssessment, PairwiseRanking, ListwiseRanking
import dspy
from dspy import Prediction
import logging 

class DataFilter(dspy.Module):
    def __init__(self, model_name, rubric_template):
        super().__init__()
        self.direct_assessment = DirectAssessment(model_name, rubric_template)
        # unsure about including pair/list wise ranking 
        self.pairwise_ranking = PairwiseRanking(model_name, rubric_template)
        self.listwise_ranking = ListwiseRanking(model_name, rubric_template)

# does not yet properly filter based on quality of response  
# should use listwise ranking - must we have 
class ResponseFilter(DataFilter):
    def forward(self, instructions, responses, reference_answers, rubric_data):
        scored_responses = []
        for instruction, response, reference_answer in zip(instructions, responses, reference_answers):
            feedback, score = self.direct_assessment.forward(instruction, response, reference_answer, rubric_data)
            if score is None:
                logging.error(f'Score is None for instruction={instruction}, response={response}')
            scored_responses.append((response, score))
  
        return scored_responses

# does not yet properly filter based on difficulty 
class DifficultyFilter(DataFilter):
    def forward(self, instructions, reference_answers, rubric_data):
        scored_instructions = []
        for instruction, reference_answer in zip(instructions, reference_answers):
            feedback, score = self.direct_assessment.forward(instruction, "", reference_answer, rubric_data)
            if score is None:
                logging.error(f'Score is None for instruction={instruction}')
            scored_instructions.append((instruction, score))

        # Filter to keep only challenging instructions
        challenging_instructions = [instr for instr, score in scored_instructions if score is not None and score > 3]

        return challenging_instructions

# does not yet properly filter based on diversity 
# should use pairwise ranking 
class DiversityFilter(DataFilter):
    def forward(self, instructions, rubric_data):
        scored_instructions = []
        for instruction in instructions:
            feedback, score = self.direct_assessment.forward(instruction, "", "", rubric_data)
            if score is None:
                logging.error(f'Score is None for instruction={instruction}')
            scored_instructions.append((instruction, feedback))
