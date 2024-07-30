import dspy
from judge import DirectAssessment
from prometheus_eval.prompts import SCORE_RUBRIC_TEMPLATE

class DataFilter(dspy.Module):
    def __init__(self):
        super().__init__()

class ResponseFilter(DataFilter):
    def __init__(self, model, score_threshold=4):
        super().__init__()
        self.judge = DirectAssessment(model=model, rubric_template=SCORE_RUBRIC_TEMPLATE)
        self.score_threshold = score_threshold
        self.rubric_data = {
            "criteria": "Evaluate the quality of the response on these three metrics: clear and accurate reply, relevant and thorough information, and polite tone.",
            "score1_description": "The response is unclear in providing an accurate reply to the instruction, is incomplete, and has an impolite tone.",
            "score2_description": "The response provides a somewhat clear and accurate reply, but is incomplete, has irrelevant information, and/or is impolite.",
            "score3_description": "The response is helpful and accurate, somewhat detailed and relevant to the instruction, with a neutral or polite tone.",
            "score4_description": "The response is overall helpful, clear and accurate. It includes many relevant details in a thorough way, and uses a polite tone.",
            "score5_description": "The response is extremely clear and helpful, with very accurate information. It includes all relevant details in a thorough way, and uses a polite tone."
        }

    def forward(self, instructions, responses):
        _, all_scores = self.judge.forward(instructions, responses, self.rubric_data, [None] * len(instructions))      
        quality_responses = []

        # list of instructions, list of list responses!
        for i, scores in enumerate(all_scores):
            response_dict = [
                {"instruction": instructions[i], "response": responses[i][j], "score": scores[j]}
                for j in range(len(scores)) if scores[j] >= self.score_threshold
            ]
            quality_responses.extend(response_dict)

        return quality_responses


class DifficultyFilter(DataFilter):
    def __init__(self, model, score_threshold):
        super().__init__()
        self.score_threshold = score_threshold
        self.judge = DirectAssessment(model=model, rubric_template=SCORE_RUBRIC_TEMPLATE)
        self.rubric_data = {
            "criteria": "Evaluate the difficulty of the instruction on these three metrics: clearly written and specific language, definitively challenging instruction in the relevant field, and creative and relevant problem-solving.",
            "score1_description": "The instruction is unclear very and too easy.",
            "score2_description": "The instruction is somewhat unclear but still easy. ",
            "score3_description": "The instruction is moderately clear and lacks challenging problem-solving.",
            "score4_description": "The instruction is clear and adequately challenging, somewhat requiring creativity or problem-solving skills.",
            "score5_description": "The instruction is very clear and highly challenging, requiring much creativity and problem-solving skills."
        }

    def forward(self, metaprompt, instructions): 
        responses = [[instr] for instr in instructions] 
        # METAPROMPT SHOULD BE STRING OR LIST???? the latter right? but trying with string...
        _, all_scores = self.judge.forward([metaprompt] * len(instructions), responses, self.rubric_data, [None] * len(instructions))
        difficult_instructions = []

        # string metaprompt, list of instructions (responses)!
        for i, scores in enumerate(all_scores):
            if scores[0] >= self.score_threshold:
                difficult_instructions.append({"instruction": instructions[i], "response": instructions[i], "score": scores[0]})
        
        return difficult_instructions