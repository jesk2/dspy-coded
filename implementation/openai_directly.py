import os
from openai import OpenAI
import dspy
from prometheus_eval.prompts import SCORE_RUBRIC_TEMPLATE

class LLMsAsJudge:
    def __init__(self, model_name, rubric_template):
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key not found in environment variables.")
        self.client = OpenAI(api_key=self.api_key)
        self.model_name = model_name
        self.rubric_template = rubric_template

    def single_absolute_grade(self, instruction, response, reference_answer, rubric_data):
        rubric = self.rubric_template.format(**rubric_data)
        feedback = self._generate_feedback(instruction, response, rubric, reference_answer)
        score = self._generate_score(instruction, response, rubric, reference_answer)
        return feedback, score

    def _generate_feedback(self, instruction, response, rubric, reference_answer):
        messages = [
            {"role": "system", "content": "You are an evaluator providing feedback based on the rubric provided."},
            {"role": "user", "content": f"{instruction}\nResponse: {response}\nRubric: {rubric}\nReference Answer: {reference_answer}\nProvide feedback based on the rubric."}
        ]
        completion = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages
        )
        feedback = completion.choices[0].message["content"]
        return feedback

    def _generate_score(self, instruction, response, rubric, reference_answer):
        messages = [
            {"role": "system", "content": "You are an evaluator providing a score based on the rubric provided."},
            {"role": "user", "content": f"{instruction}\nResponse: {response}\nRubric: {rubric}\nReference Answer: {reference_answer}\nProvide a score based on the rubric."}
        ]
        completion = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages
        )
        score = completion.choices[0].message["content"]
        return score

class DirectAssessment(LLMsAsJudge):
    def forward(self, instruction, response, reference_answer, rubric_data):
        feedback, score = self.single_absolute_grade(instruction, response, reference_answer, rubric_data)
        return dspy.Prediction(feedback=feedback, score=score)

# Testing 
if __name__ == "__main__":
    rubric_template = SCORE_RUBRIC_TEMPLATE
    judge_model_name = 'gpt-3.5-turbo'

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

    direct_assessment = DirectAssessment(
        model_name=judge_model_name,
        rubric_template=rubric_template
    )
    result = direct_assessment.forward(instruction, response, reference_answer, rubric_data)
    print("Feedback:", result.feedback)
    print("Score:", result.score)
