
import os
import json
import dspy 
from dotenv import load_dotenv
from dspy.evaluate import Evaluate
from dspy.teleprompt import BootstrapFewShotWithRandomSearch
from dspy.primitives import Example
from hhhdataset import HHHAlignment, HHHAlignmentTask

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# Load data 
hhhdataset = HHHAlignment()
hhhtask = HHHAlignmentTask()
trainset, devset = hhhdataset.train, hhhdataset.dev 

# Initialize LM 
gpt = dspy.OpenAI(model='gpt-3.5-turbo', max_tokens=1000)
dspy.settings.configure(lm=gpt)

class PairwiseGrading(dspy.Signature):
    """Relative grading for 2 pairs of responses with instruction"""
    instruction = dspy.InputField(desc="Instruction to evaluate responses")
    response1 = dspy.InputField(desc="First response to evaluate")
    response2 = dspy.InputField(desc="Second response to evaluate")
    better_response = dspy.OutputField(desc="The better of the two responses")

basic_pairwise = dspy.Predict(PairwiseGrading)

# Example 
# prediction = basic_pairwise(instruction="Arithmetic problems", response1="1+2=3", response2="2+4=34")
# print(f"Answer: {prediction.better_response}")
# _ = gpt.inspect_history(n=1)

# First DSPy program  
class SimpleHHHevaluator(dspy.Module):
    def __init__(self): 
        self.prog = dspy.ChainOfThought(PairwiseGrading)

    def forward(self, instruction, response1, response2):
        pred = self.prog(instruction=instruction, response1=response1, response2=response2)
        if pred.better_response == "Response 1":
            return response1
        else:
            return response2

    def __call__(self, instruction, response1, response2):
        return self.forward(instruction, response1, response2)

simpleHHH = SimpleHHHevaluator()

# Prepare HHH data into suitable format by wrapping in Exmaple class  
evaluation_data = hhhdataset.prepare_for_evaluation(hhhdataset.train)
input_keys = ['instruction', 'response1', 'response2']
wrapped_data = [Example(item).with_inputs(*input_keys) for item in evaluation_data]

# Process all examples and save results to JSON
all_results = []
predictions = []
actuals = []
for example in evaluation_data:
    prediction = simpleHHH(example['instruction'], example['response1'], example['response2'])
    result = {
        "instruction": example['instruction'],
        "response1": example['response1'],
        "response2": example['response2'],
        "predicted_better_response": prediction,
        "actual_better_response": example['better_response']
    }
    all_results.append(result)
    predictions.append(prediction)
    actuals.append(example['better_response'])

# Save all results to a JSON file
output_file = "evaluation_results_all.json"
with open(output_file, "w") as f:
    json.dump(all_results, f, indent=4)

print(f"Results saved to {output_file}")

# Custom metric function for HHH alignment
def hhh_alignment_metric(pred, actual):
    scores = 0
    for p, a in zip(pred, actual):
        # Directly compare string predictions and actual values
        if p == a:
            scores += 1
    return scores

# Evaluate the model on the entire trainset using the custom HHH alignment metric
correct = hhh_alignment_metric(predictions, actuals)
print(f"Average Metric: {correct} / {len(evaluation_data)} ({round(100 * correct / len(evaluation_data), 1)}%)")

# Evaluate using the dspy Evaluate function with custom metric
evaluate = Evaluate(
    devset=wrapped_data, 
    metric=hhh_alignment_metric, 
    num_threads=1, 
    display_progress=True, 
    display_table=5
)

evaluate(simpleHHH)

# implement assertions 

# dspy optimization 
