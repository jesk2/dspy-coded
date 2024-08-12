import dspy
import random
import tqdm
from datasets import load_dataset
from collections import defaultdict

class HHHAlignment:
    def __init__(self) -> None:
        super().__init__()

        # Load the HHH_alignment dataset from Huggingface
        dataset = load_dataset("HuggingFaceH4/hhh_alignment", "helpful")
        
        hf_data = dataset['test']
        grouped_data = defaultdict(list)

        # Group the data by instruction
        for example in tqdm.tqdm(hf_data):
            question = example['input']
            choices = example['targets']['choices']
            labels = example['targets']['labels']

            for choice, label in zip(choices, labels):
                grouped_data[question].append((choice, label))

        official_data = []

        # Process the grouped data to ensure we get pairs of responses
        for question, responses in grouped_data.items():
            if len(responses) >= 2:  # Ensure we have at least one pair
                for i in range(0, len(responses) - 1, 2):
                    if i + 1 < len(responses):
                        response_pair = (responses[i][0], responses[i + 1][0])
                        label_pair = (responses[i][1], responses[i + 1][1])
                        official_data.append(dict(question=question, responses=response_pair, labels=label_pair))

        rng = random.Random(0)
        rng.shuffle(official_data)

        trainset = official_data[:200]
        devset = official_data[200:500]
        testset = official_data[500:]

        trainset = [dspy.Example(**x).with_inputs('input') for x in trainset]
        devset = [dspy.Example(**x).with_inputs('input') for x in devset]
        testset = [dspy.Example(**x).with_inputs('input') for x in testset]

        self.train = trainset
        self.dev = devset
        self.test = testset


    def prepare_for_evaluation(self, dataset):
        evaluation_data = []
        for example in dataset:
            question = example['question']
            response1, response2 = example['responses']
            label1, label2 = example['labels']
            better_response = response1 if label1 > label2 else response2
            evaluation_data.append({
                "instruction": question,
                "response1": response1,
                "response2": response2,
                "better_response": better_response
            })
        return evaluation_data

# Step 3: Create Test Cases for the Judge and Ranking Classes
class HHHAlignmentTask:
    def __init__(self):
        hhhdataset = HHHAlignment()

        self.trainset = hhhdataset.train
        self.devset = hhhdataset.dev
        self.testset = hhhdataset.test

    def get_program(self):
        # Define the signature for the task
        return dspy.ChainOfThought("question -> choice")
    
    def get_metric(self, predictions, references):
        # Implement a metric function to calculate the accuracy
        correct = sum(1 for pred, ref in zip(predictions, references) if pred == ref)
        total = len(references)
        return correct, total