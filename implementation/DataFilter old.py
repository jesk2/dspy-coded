import dspy
import datasets
from dspy import Prediction
from dspy.evaluate import Evaluate
from dspy.evaluate.metrics import answer_exact_match, answer_passage_match
from prometheus_eval import PrometheusEval
from prometheus_eval.mock import MockLLM
from prometheus_eval.prompts import ABSOLUTE_PROMPT, SCORE_RUBRIC_TEMPLATE, RELATIVE_PROMPT
import logging

logging.basicConfig(level=logging.DEBUG)

# DOES NOT IMPLEMENT PROMETHEUS / uses local examples 

# Step 1: Select a Hugging Face Dataset (Example: IMDb sentiment dataset)
dataset = datasets.load_dataset('imdb')

# Step 2: Convert Dataset to Dictionary Format
def convert_to_dict(example):
    return {
        'instruction': '',  # Define based on your dataset structure
        'response': example['text'],  # Assuming 'text' field contains response
    }

def mock_program(instruction, response):
    # Simulate model prediction using MockLLM
    mock_llm = MockLLM()
    prediction = {'response': 'This is a sample response.'}
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

# Use PrometheusEval for scoring responses
def custom_metric(example, prediction):
    # Initialize PrometheusEval with the mock model for scoring
    prometheus_eval = PrometheusEval(model=MockLLM(), absolute_grade_template=ABSOLUTE_PROMPT)

    rubric_data = {
        "criteria": "Is the model proficient in applying empathy and emotional intelligence to its responses when the user conveys emotions or faces challenging circumstances?",
        "score1_description": "The model neglects to identify or react to the emotional tone of user inputs, giving responses that are unfitting or emotionally insensitive.",
        "score2_description": "The model intermittently acknowledges emotional context but often responds without sufficient empathy or emotional understanding.",
        "score3_description": "The model typically identifies emotional context and attempts to answer with empathy, yet the responses might sometimes miss the point or lack emotional profundity.",
        "score4_description": "The model consistently identifies and reacts suitably to emotional context, providing empathetic responses. Nonetheless, there may still be sporadic oversights or deficiencies in emotional depth.",
        "score5_description": "The model excels in identifying emotional context and persistently offers empathetic, emotionally aware responses that demonstrate a profound comprehension of the user's emotions or situation."
    }

    score_rubric = SCORE_RUBRIC_TEMPLATE.format(**rubric_data)

    try:
        # Use PrometheusEval to get the score for the response
        feedback, score = prometheus_eval.single_absolute_grade(
            instruction=example['instruction'],
            response=prediction['response'],
            rubric=score_rubric
        )
        if score is None:
            raise ValueError("Received None as score")
        print(f"Feedback: {feedback}, Score: {score}")
        return score

    except Exception as e:
        print(f"Error in custom_metric: {e}")
        return 5  # Return a default score of 0 in case of an error


# Step 4: Use Evaluator with Custom Metric
evaluator = Evaluate(devset=data_train, metric=custom_metric, num_threads=1, display_progress=True, display_table=5)
results = evaluator(program=mock_program, metric=custom_metric, return_all_scores=True)

# Step 5: DataFilter Class for Generic Filtering
class DataFilter(dspy.Module):
    def __init__(self, data, results):
        super().__init__()
        self.data = data
        self.results = results 

    def filter(self):
        print("Results inside filter method:", self.results)
        average_metric, scores = self.results  # Unpack the results to get scores

        # Implement specific filtering logic in subclasses
        raise NotImplementedError("Subclasses should implement the filter method.")

# Step 6: DiversityFilter for Diversity Filtering
class DiversityFilter(DataFilter):
    def filter(self):
        print("Applying DiversityFilter...")
        filtered_data = []

        for item, score in zip(self.data, self.results[1]):
            # Example logic: Check if the response has diverse vocabulary
            if len(set(item['response'].split())) > 5:  # Example diversity condition
                filtered_data.append(item)

        return filtered_data


# Step 7: DifficultyFilter for Difficulty Filtering
class DifficultyFilter(DataFilter):
    def filter(self):
        print("Applying DifficultyFilter...")
        filtered_data = []

        for item, score in zip(self.data, self.results[1]):
            # Example logic: Check if the instruction is complex
            if len(item['instruction'].split()) > 5:  # Example difficulty condition
                filtered_data.append(item)

        return filtered_data

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

    # Normalize the data
    normalized_data = normalize_data(data)

    # Use Evaluate to get results
    evaluator = Evaluate(devset=normalized_data, metric=custom_metric, num_threads=1, display_progress=True, display_table=5)
    results = evaluator(program=mock_program, metric=custom_metric, return_all_scores=True)

    # Instantiate and use the filters
    diversity_filter = DiversityFilter(normalized_data, results)
    filtered_data_diversity = diversity_filter.filter()

    difficulty_filter = DifficultyFilter(normalized_data, results)
    filtered_data_difficulty = difficulty_filter.filter()

    print("Filtered data (Diversity):", filtered_data_diversity)
    print("Filtered data (Difficulty):", filtered_data_difficulty)

