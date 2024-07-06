# just LLMsAsJudge and DirectAssessment classes 

import dspy
from dspy import Prediction
from dspy.evaluate.metrics import answer_exact_match, answer_passage_match
from prometheus_eval import PrometheusEval
from prometheus_eval.mock import MockLLM
from prometheus_eval.prompts import ABSOLUTE_PROMPT, SCORE_RUBRIC_TEMPLATE, RELATIVE_PROMPT
import logging 
# for testing use from prometheus_eval.mock import MockLLM (result is Hello by default and judge_model = 'absolute' 

logging.basicConfig(level=logging.DEBUG)

class LLMsAsJudge(dspy.Module):
    def __init__(self, model_name, rubric_template):
        super().__init__()
        self.judge = PrometheusEval(model=MockLLM(model_name))
        self.rubric_template = rubric_template

    # def forward(self, instruction, response, reference_answer, rubric_data):
    #     rubric = self.rubric_template.format(**rubric_data)
    #     logging.debug(f"Formatted rubric: {rubric}")
        
    #     try:
    #         feedback, score = self.judge.single_absolute_grade(
    #             instruction=instruction,
    #             response=response,
    #             rubric=rubric,
    #             reference_answer=reference_answer
    #         )
    #         logging.debug(f"Received feedback: {feedback}, score: {score}")
    #     except Exception as e:
    #         logging.error(f"Error in single_absolute_grade: {e}")
    #         feedback, score = None, None
        
    #     return Prediction(feedback=feedback, score=score)

class DirectAssessment(LLMsAsJudge):
    def forward(self, instruction, response, reference_answer, rubric_data):
        return
        # return super().forward(instruction, response, reference_answer, rubric_data)
    
class PairwiseRanking(LLMsAsJudge):
    def forward(self, instruction, responseA, responseB, reference_answer, rubric_data):
        return
        # return super().forward(instruction, response, reference_answer, rubric_data)
    
class ListwiseRanking(LLMsAsJudge):
    def forward(self, instruction, response_list, reference_answer, rubric_data):
        return
        # return super().forward(instruction, response, reference_answer, rubric_data)

# testing 
if __name__ == "__main__":
    rubric_template = SCORE_RUBRIC_TEMPLATE
    judge_model = 'absolute' 

    instruction = "Struggling with a recent break-up, a person opens up about the intense feelings of loneliness and sadness. They ask for advice on how to cope with the heartbreak and move forward in life."
    response = "It's important to allow yourself to feel your emotions and give yourself time to heal. Try to focus on self-care and seek support from friends and family."
    reference_answer = "To cope with heartbreak, you can start by acknowledging your emotions, spending time with loved ones, engaging in activities you enjoy, and considering professional help if needed."
    rubric_data = {
        "criteria": "Is the model proficient in applying empathy and emotional intelligence to its responses when the user conveys emotions or faces challenging circumstances?",
        "score1_description": "The model neglects to identify or react to the emotional tone of user inputs, giving responses that are unfitting or emotionally insensitive.",
        "score2_description": "The model intermittently acknowledges emotional context but often responds without sufficient empathy or emotional understanding.",
        "score3_description": "The model typically identifies emotional context and attempts to answer with empathy, yet the responses might sometimes miss the point or lack emotional profundity.",
        "score4_description": "The model consistently identifies and reacts suitably to emotional context, providing empathetic responses. Nonetheless, there may still be sporadic oversights or deficiencies in emotional depth.",
        "score5_description": "The model excels in identifying emotional context and persistently offers empathetic, emotionally aware responses that demonstrate a profound comprehension of the user's emotions or situation."
    }

    gpt3 = dspy.OpenAI(
        model="gpt-3.5-turbo",
        max_tokens=4000,
        model_type="chat",
    )

    with dspy.context(lm=gpt3):
        direct_assessment = DirectAssessment(
            model_name=judge_model,
            rubric_template=rubric_template
        )
        result = direct_assessment.forward(instruction, response, reference_answer, rubric_data)
        # print("Feedback:", result.feedback)
        # print("Score:", result.score)
