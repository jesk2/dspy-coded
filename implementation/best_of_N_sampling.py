# Data Filtering 
from judge import ListwiseRanking
import dspy
from dspy import Prediction

class BestofNSampling(dspy.Module):
    def __init__(self):
        super().__init__()
        self.judge = ListwiseRanking()
        
    def forward(self):
        return