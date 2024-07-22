# Data Filtering 
from judge import ListwiseRanking
import dspy
from dspy import Prediction

class BestofNSampling(dspy.Module):
    def __init__(self, model, rubric_template):
        super().__init__()
        self.judge = ListwiseRanking(model=model, rubric_template=rubric_template)
        
    def forward(self, instructions, response_list, reference_answers, rubric_data, num):
        all_sorted_responses = self.judge.forward(instructions, response_list, reference_answers, rubric_data)
        top_n = []
        
        for response in all_sorted_responses:
            top_n.append(response[:num])
        
        return top_n