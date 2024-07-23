from judge import ListwiseRanking
import dspy
from prometheus_eval.prompts import SCORE_RUBRIC_TEMPLATE

class DataFilter(dspy.Module):
    def __init__(self):
        super().__init__()

class ResponseFilter(DataFilter):
    def __init__(self, model):
        super().__init__()
        self.judge = ListwiseRanking(model=model, rubric_template=SCORE_RUBRIC_TEMPLATE)
        self.rubric_data = {
            "criteria": "Is the response coherent with regards to the instruction, factually accurate, logically robust, and insightful?",
            "score1_description": "The response is very incoherent with regards to the instruction, and factually inaccurate, not logically robust, and not insightful.",
            "score2_description": "The response is mostly incoherent with regards to the instruction, and is mostly factually inaccurate, not logically robust, and/or insightful. ",
            "score3_description": "The response is somewhat coherent with regards to the instruction, and is somewhat either factually accurate, logically robust, and/or insightful.",
            "score4_description": "The response is overall coherent with regards to the instruction, and is generally factually accurate, logically robust, and/or insightful.",
            "score5_description": "The response is extremely coherent with regards to the instruction, factually accurate, logically robust, and provides insightful information."
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
            "criteria": "Is the instruction challenging in the relevant field and does it use clear and specific language?",
            "score1_description": "The instruction is too easy and uses very unclear and generic language.",
            "score2_description": "The instruction is still easy and generally uses unclear or generic language.",
            "score3_description": "The instruction is somewhat challenging but does not require much creativity nor problem-solving, with moderately clear and specific language.",
            "score4_description": "The instruction is appropriately challenging, somewhat requiring either creativity or problem-solving skills, and mostly uses clear and specific language.",
            "score5_description": "The instruction is highly challenging, requiring much creativity and problem-solving skills, and uses very clear and specific language."
        }


    def forward(self, instructions):
        # Placeholder responses and reference_answers for difficulty evaluation
        responses = [[instr] for instr in instructions]  # each instruction is its own "response"
        reference_answers = [None] * len(instructions)
        # Get sorted instructions
        sorted_instructions = self.judge.forward(instructions, responses, reference_answers, self.rubric_data)
        # Return sorted instructions
        return sorted_instructions