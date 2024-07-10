import dspy
from prometheus_eval.prompts import ABSOLUTE_PROMPT, SCORE_RUBRIC_TEMPLATE, RELATIVE_PROMPT
import logging 

logging.basicConfig(level=logging.DEBUG)

class LLMsAsJudge(dspy.Module):
    def __init__(self, model, rubric_template):
        super().__init__()
        self.model = model
        self.rubric_template = rubric_template

class DirectAssessment(LLMsAsJudge):
    def forward(self, instruction, response, reference_answer, rubric_data):
        score_rubric = self.rubric_template.format(**rubric_data)
        # first proceed with single absolute grade 
        feedback, score = self.model.single_absolute_grade(
            instruction=instruction,
            response=response,
            rubric=score_rubric,
            reference_answer=reference_answer
        )
        # logging.debug(f'DirectAssessment forward: instruction={instruction}, response={response}, score={score}')
        return feedback, score

class PairwiseRanking(LLMsAsJudge):
    def forward(self, instruction, responseA, responseB, reference_answer, rubric_data):
        pass 

class ListwiseRanking(LLMsAsJudge):
    def forward(self, instruction, response_list, reference_answer, rubric_data):
        pass 