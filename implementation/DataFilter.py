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
            "criteria": "Is the response helpful, thorough, and polite?",
            "score1_description": "The response is unhelpful, incomplete, and impolite.",
            "score2_description": "The response is somewhat helpful but lacks completeness and politeness.",
            "score3_description": "The response is helpful, somewhat thorough, and polite.",
            "score4_description": "The response is very helpful, thorough, and polite.",
            "score5_description": "The response is extremely helpful, thorough, and polite."
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
            "criteria": "Is the instruction clear and appropriately challenging?",
            "score1_description": "The instruction is unclear and too easy.",
            "score2_description": "The instruction is somewhat clear but too easy.",
            "score3_description": "The instruction is clear but lacks challenge.",
            "score4_description": "The instruction is clear and appropriately challenging.",
            "score5_description": "The instruction is very clear and highly challenging."
        }

    def forward(self, instructions):
        # Placeholder responses and reference_answers for difficulty evaluation
        responses = [[instr] for instr in instructions]  # each instruction is its own "response"
        reference_answers = [None] * len(instructions)
        # Get sorted instructions
        sorted_instructions = self.judge.forward(instructions, responses, reference_answers, self.rubric_data)
        # Return sorted instructions
        return sorted_instructions