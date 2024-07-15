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
    def forward(self, instructions, responses, reference_answers, rubric_data):
        rubric = self.rubric_template.format(**rubric_data)
        feedbacks, score = self.model.absolute_grade(
            instructions=instructions,
            responses=responses,
            rubric=rubric,
            reference_answers=reference_answers
        )
        # logging.debug(f'DirectAssessment forward: instruction={instruction}, response={response}, score={score}')
        return feedbacks, score 

class PairwiseRanking(LLMsAsJudge):
    def forward(self, instructions, responseA, responseB, reference_answer, rubric_data):
        rubric = self.rubric_template.format(**rubric_data)
        feedbacks, score = self.model.relative_grade(
            instructions = instructions,
            responses_A = responseA,
            responses_B = responseB,
            rubric = rubric, 
            reference_answers = reference_answer
        )
        # logging.debug(f'PairwiseRanking forward: instruction={instruction}, responseA={responseA}, responseB={responseB}, score={score}')
        return feedbacks, score

class ListwiseRanking(PairwiseRanking):
    def forward(self, instruction, response_list, reference_answer, rubric_data):
        pass 