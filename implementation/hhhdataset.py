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

# Instantiate the class to prepare the data
hhhdataset = HHHAlignment()
