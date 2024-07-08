import dspy
from dspy import Prediction
from dspy.evaluate.metrics import answer_exact_match, answer_passage_match
from prometheus_eval import PrometheusEval
from prometheus_eval.mock import MockLLM
from prometheus_eval.prompts import ABSOLUTE_PROMPT, SCORE_RUBRIC_TEMPLATE, RELATIVE_PROMPT
import logging 

logging.basicConfig(level=logging.DEBUG)

class LLMsAsJudge(dspy.Module):
    def __init__(self, model_name, rubric_template):
        super().__init__()
        self.judge = PrometheusEval(model=MockLLM(), absolute_grade_template=ABSOLUTE_PROMPT)
        self.rubric_template = rubric_template

class DirectAssessment(LLMsAsJudge):
    def forward(self, instruction, response, reference_answer, rubric_data):
        score_rubric = self.rubric_template.format(**rubric_data)
        feedback, score = self.judge.single_absolute_grade(
            instruction=instruction,
            response=response,
            rubric=score_rubric,
            reference_answer=reference_answer
        )
        logging.debug(f'DirectAssessment forward: instruction={instruction}, response={response}, score={score}')
        return feedback, score

# unsure about this, for comparing pairs of responses
class RelativeAssessment(LLMsAsJudge):
    def forward(self, instruction, responseA, responseB, reference_answer, rubric_data):
        pass 

class PairwiseRanking(LLMsAsJudge):
    def forward(self, instruction, responseA, responseB, reference_answer, rubric_data):
        score_rubric = self.rubric_template.format(**rubric_data)
        feedback, score = self.judge.relative_grade(
            instruction=instruction,
            responseA=responseA,
            responseB=responseB,
            rubric=score_rubric,
            reference_answer=reference_answer
        )
        return feedback, score

class ListwiseRanking(LLMsAsJudge):
    def forward(self, instruction, response_list, reference_answer, rubric_data):
        score_rubric = self.rubric_template.format(**rubric_data)
        feedback, scores = self.judge.listwise_grade(
            instruction=instruction,
            responses=response_list,
            rubric=score_rubric,
            reference_answer=reference_answer
        )
        return feedback, scores