from judge import ListwiseRanking
import dspy
from prometheus_eval.prompts import SCORE_RUBRIC_TEMPLATE

# Data Filtering
class DataFilter(dspy.Module):
    def __init__(self):
        super().__init__()

class ResponseFilter(DataFilter):
    def __init__(self, model):
        super().__init__()
        self.judge = ListwiseRanking(model=model, rubric_template=SCORE_RUBRIC_TEMPLATE)
        self.rubric_data = {
            "criteria": "Evaluate the quality of the response on these three metrics: clear and accurate reply, relevant and thorough information, and polite tone.",
            "score1_description": "The response is unclear in providing an accurate reply to the instruction, is incomplete, and has an impolite tone.",
            "score2_description": "The response provides a somewhat clear and accurate reply, but is incomplete, has irrelevant information, and/or is impolite.",
            "score3_description": "The response is helpful and accurate, somewhat detailed and relevant to the instruction, with a neutral or polite tone.",
            "score4_description": "The response is overall helpful, clear and accurate. It includes many relevant details in a thorough way, and uses a polite tone.",
            "score5_description": "The response is extremely clear and helpful, with very accurate information. It includes all relevant details in a thorough way, and uses a polite tone."
        }

    def forward(self, instructions, responses, reference_answers, num_responses=5):
        assert all(isinstance(r, list) for r in responses), "responses must be a list of lists"
        sorted_responses = self.judge.forward(instructions, responses, reference_answers, self.rubric_data)
        top_responses = [responses[i][:num_responses] for i in range(len(sorted_responses))]
        return top_responses
    
class DifficultyFilter(DataFilter):
    def __init__(self, model):
        super().__init__()
        self.judge = ListwiseRanking(model=model, rubric_template=SCORE_RUBRIC_TEMPLATE)
        self.rubric_data = {
            "criteria": "Evaluate the difficulty of the instruction on these three metrics: clearly written and specific language, definitively challenging instruction in the relevant field, and creative and relevant problem-solving.",
            "score1_description": "The instruction is unclear very and too easy.",
            "score2_description": "The instruction is somewhat unclear but still easy. ",
            "score3_description": "The instruction is moderately clear and lacks challenging problem-solving.",
            "score4_description": "The instruction is clear and adequately challenging, somewhat requiring creativity or problem-solving skills.",
            "score5_description": "The instruction is very clear and highly challenging, requiring much creativity and problem-solving skills."
        }

    def forward(self, instructions):
        # Placeholder responses and reference_answers for difficulty evaluation
        responses = [[instr] for instr in instructions]  # each instruction is its own "response"
        reference_answers = [None] * len(instructions)
        # Get sorted instructions
        sorted_instructions = self.judge.forward(instructions, responses, reference_answers, self.rubric_data)
        # Return sorted instructions
        return sorted_instructions