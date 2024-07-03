# just LLMsAsJudge and DirectAssessment classes 

import dspy
import datasets
from dspy import Prediction
from dspy.evaluate import Evaluate
from dspy.evaluate.metrics import answer_exact_match, answer_passage_match
from prometheus_eval import PrometheusEval
from prometheus_eval.mock import MockLLM
from prometheus_eval.prompts import ABSOLUTE_PROMPT, SCORE_RUBRIC_TEMPLATE, RELATIVE_PROMPT
import logging 
# for testing use from prometheus_eval.mock import MockLLM (result is Hello by default and judge_model = 'absolute' 

logging.basicConfig(level=logging.DEBUG)


# Step 1: Select a Hugging Face Dataset (Example: IMDb sentiment dataset)
dataset = datasets.load_dataset('imdb')

# Step 2: Convert Dataset to Dictionary Format
def convert_to_dict(example):
    return {
        'instruction': '',  # Define based on your dataset structure
        'response': example['text'],  # Assuming 'text' field contains response
    }

def mock_program(instruction, response):
    # Simulate model prediction; for example purposes, we'll just echo the response
    prediction = {'response': 'This is a mock response'}
    return prediction

class ExampleWrapper(dict):
    def __init__(self, example):
        super().__init__(example)
        self.example = example

    def inputs(self):
        return self.example

# Step 2: Ensure dataset structure and access correct field
data_train = [ExampleWrapper({'instruction': '', 'response': ex['text']}) for ex in dataset['train'].select(range(100))]

# Step 3: Define a Custom Metric Function
def custom_metric(example, prediction):
    # Define your metric calculation here based on example and prediction
    return 1.0  # Placeholder, replace with actual metric calculation

# Step 4: Use Evaluator with Custom Metric
evaluator = Evaluate(devset=data_train, metric=custom_metric, num_threads=1, display_progress=True, display_table=5)
results = evaluator(program=mock_program, metric=custom_metric, return_all_scores=True)

# Data filtering DOES NOT WORK THE WAY IT SHOULD!!!!!!!!

# Step 5: Further Processing with DataFilter
class DataFilter(dspy.Module):
    def __init__(self, data, results):
        super().__init__()
        self.data = data
        self.results = results 

    def filter(self):
        print("Results inside filter method:", self.results)
        average_metric, scores = self.results  # Unpack the results to get scores
        filtered_data = [item for item, score in zip(self.data, scores) if score > 0.5]  # Example filtering condition
        return filtered_data

class ResponseFilter(DataFilter):
    def __init__(self, data):
        super().__init__(data, self.quality_metric)

    def quality_metric(self, example, pred, trace=None):
        return "good" in pred['response'].lower()

class DifficultyFilter(DataFilter):
    def __init__(self, data):
        super().__init__(data, self.difficulty_metric)

    def difficulty_metric(self, example, pred, trace=None):
        return len(example['instruction'].split()) > 5

class DiversityFilter(DataFilter):
    def __init__(self, data):
        super().__init__(data, self.diversity_metric)

    def diversity_metric(self, example, pred, trace=None):
        return len(set(pred['response'].split())) > 10

def normalize_data(data):
    normalized_data = []
    for item in data:
        if isinstance(item, dict):
            if 'instruction' in item and 'response' in item:
                normalized_data.append(item)
            else:
                raise ValueError("Dictionary items must contain 'instruction' and 'response' keys.")
        else:
            raise TypeError("Each item in the dataset must be a dictionary.")
    return normalized_data


if __name__ == "__main__":
    # Assuming `data` is your dataset (already in dictionary format)
    data = [
        {'instruction': 'What is the capital of France?', 'response': 'The capital of France is Paris.'},
        {'instruction': 'What is the capital of Singapore?', 'response': 'The capital of Singapore is Singapore.'},
        {'instruction': 'Describe a beautiful sunset.', 'response': 'The sunset is beautiful with hues of orange and pink.'},
    ]

    # Instantiate and use the filters
    data_filter = DataFilter(data, results)
    filtered_data = data_filter.filter()

    print("Filtered data:", filtered_data)