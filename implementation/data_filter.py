# Data Filtering 
from judge import DirectAssessment
import dspy
from dspy import Prediction

class DataFilter(dspy.Module):
    def __init__(self):
        super().__init__()
        self.judge = DirectAssessment()
        


class ResponseFilter(DataFilter):
    def forward(self, instructions, responses):
        for i,r in zip(instructions,responses):
            pass

        return

class DifficultyFilter(DataFilter):
    def forward(self, instructions):
        for i,r in zip(instructions):
            pass

        return
    
class DiversityFilter(DataFilter):
    def forward(self, instructions):
        for i,r in zip(instructions):
            pass

        return
    
