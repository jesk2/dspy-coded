# DOES NOT WORK AS OF NOW!!!!

import dspy
from prometheus_eval.litellm import LiteLLM
from prometheus_eval import PrometheusEval
from prometheus_eval.prompts import ABSOLUTE_PROMPT, SCORE_RUBRIC_TEMPLATE, RELATIVE_PROMPT

class LLMsAsJudge(dspy.Signature):
    model_name = dspy.InputField()
    rubric_template = dspy.InputField()
    instruction = dspy.InputField()
    response = dspy.InputField()
    reference_answer = dspy.InputField()
    rubric_data = dspy.InputField()

    feedback = dspy.OutputField()
    score = dspy.OutputField()

    def single_absolute_grade(self, instruction, response, reference_answer, rubric_data):
        rubric = self.rubric_template.format(**rubric_data)
        feedback, score = self.judge.single_absolute_grade(
            instruction=instruction,
            response=response,
            rubric=rubric,
            reference_answer=reference_answer
        )
        return feedback, score


class DirectAssessment(LLMsAsJudge):
    def __init__(self, **data):
        super().__init__(**data)

    def forward(self, instruction, response, reference_answer, rubric_data):
        feedback, score = self.single_absolute_grade(instruction, response, reference_answer, rubric_data)
        return dspy.Prediction(feedback=feedback, score=score)

class PairwiseRanking(dspy.Signature):
    model_name = dspy.InputField()
    relative_template = dspy.InputField()
    instruction = dspy.InputField()
    response_A = dspy.InputField()
    response_B = dspy.InputField()
    reference_answer = dspy.InputField()
    rubric = dspy.InputField()

    feedback = dspy.OutputField()
    score = dspy.OutputField()

    def __init__(self, model_name, relative_template):
        super().__init__()
        self.model = LiteLLM(model_name)
        self.judge = PrometheusEval(model=self.model)
        self.relative_template = relative_template

    def forward(self, instruction, response_A, response_B, reference_answer, rubric):
        data = {
            "instruction": instruction,
            "response_A": response_A,
            "response_B": response_B,
            "reference_answer": reference_answer,
            "rubric": rubric
        }
        feedback, score = self.judge.single_relative_grade(**data)
        return dspy.Prediction(feedback=feedback, score=score)


class ListwiseRanking(dspy.Signature):
    model_name = dspy.InputField()
    listwise_template = dspy.InputField()
    instruction = dspy.InputField()
    responses = dspy.InputField()
    reference_answer = dspy.InputField()
    rubric = dspy.InputField()

    feedbacks = dspy.OutputField()
    scores = dspy.OutputField()

    def __init__(self, model_name, listwise_template):
        super().__init__()
        self.model = LiteLLM(model_name)
        self.judge = PrometheusEval(model=self.model)
        self.listwise_template = listwise_template

    def forward(self, instruction, responses, reference_answer, rubric):
        feedbacks, scores = self.judge.listwise_grade(
            instruction=instruction,
            responses=responses,
            rubric=rubric,
            reference_answer=reference_answer
        )
        return dspy.Prediction(feedbacks=feedbacks, scores=scores)


class SFTDatasetConstruction(dspy.Module):
    def __init__(self, judge_model_name, rubric_template):
        super().__init__()
        self.judge = DirectAssessment(judge_model_name, rubric_template)

    def forward(self, dataset, rubric_data):
        filtered_data = []
        for data in dataset:
            instruction, response, reference_answer = data
            result = self.judge(instruction=instruction, response=response, reference_answer=reference_answer, rubric_data=rubric_data)
            if result.score >= 4:  # Example threshold
                filtered_data.append(data)
        return dspy.Prediction(filtered_data=filtered_data)

class SFTDataAugmentation(dspy.Module):
    def __init__(self, model_name):
        super().__init__()
        self.model = LiteLLM(model_name)

    def forward(self, dataset, num_augmentations=5):
        augmented_data = []
        for data in dataset:
            instruction, response = data
            for _ in range(num_augmentations):
                new_response = self.model.generate_response(instruction)
                augmented_data.append((instruction, new_response))
        return dspy.Prediction(augmented_data=augmented_data)


class DPODatasetConstruction(SFTDatasetConstruction):
    def forward(self, dataset, rubric_data):
        good_responses, bad_responses = [], []
        for data in dataset:
            instruction, response, reference_answer = data
            result = self.judge(instruction=instruction, response=response, reference_answer=reference_answer, rubric_data=rubric_data)
            if result.score >= 4:
                good_responses.append(data)
            else:
                bad_responses.append(data)
        return dspy.Prediction(good_responses=good_responses, bad_responses=bad_responses)

class DPODataAugmentation(SFTDataAugmentation):
    def forward(self, dataset, num_augmentations=5):
        augmented_data = super().forward(dataset, num_augmentations).augmented_data
        bad_data = []
        for data in augmented_data:
            instruction, response = data
            bad_response = self.model.generate_bad_response(instruction)
            bad_data.append((instruction, bad_response))
        return dspy.Prediction(augmented_data=augmented_data, bad_data=bad_data)


class BestofNSampling(dspy.Module):
    def __init__(self, model_name, judge_model_name, rubric_template):
        super().__init__()
        self.model = LiteLLM(model_name)
        self.judge = DirectAssessment(judge_model_name, rubric_template)

    def forward(self, instruction, rubric_data, num_responses=5):
        responses = [self.model.generate_response(instruction) for _ in range(num_responses)]
        best_score = -1
        best_response = None
        for response in responses:
            result = self.judge(instruction=instruction, response=response, reference_answer=None, rubric_data=rubric_data)
            if result.score > best_score:
                best_score = result.score
                best_response = response
        return dspy.Prediction(best_response=best_response)


if __name__ == "__main__":
    rubric_template = SCORE_RUBRIC_TEMPLATE  # Define your rubric template here
    judge_model_name = 'openai/prometheus-eval/prometheus-7b-v2.0'

    # Example data
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

    # Initialize and use DirectAssessment
    gpt3 = dspy.OpenAI(
        model="gpt-3.5-turbo",
        max_tokens=4000,
        model_type="chat",
        api_base="http://0.0.0.0:4000/",
    )

    with dspy.context(lm=gpt3):
        direct_assessment = DirectAssessment(
            model_name=judge_model_name,
            rubric_template=rubric_template,
            instruction=instruction,
            response=response,
            reference_answer=reference_answer,
            rubric_data=rubric_data
        )
        result = direct_assessment.forward(instruction, response, reference_answer, rubric_data)
        print("Feedback:", result.feedback)
        print("Score:", result.score)
