# BUILD A RAG PIPELINE

import sys
import os
import dspy
import warnings
from urllib3.exceptions import NotOpenSSLWarning

warnings.filterwarnings('ignore', category=NotOpenSSLWarning)

################
#   LM Setup   #
################

# Initialize the OpenAI client with the API key from environment variable
import openai 

# api_key =""
# openai.api_key = api_key

turbo = dspy.OpenAI(model='gpt-3.5-turbo')
colbertv2_wiki17_abstracts = dspy.ColBERTv2(url='http://20.102.90.50:2017/wiki17_abstracts')

dspy.settings.configure(lm=turbo, rm=colbertv2_wiki17_abstracts)

###########################
#   Loading the dataset   #
###########################

from dspy.datasets import HotPotQA

dataset = HotPotQA(train_seed=1, train_size=20, eval_seed=2023, dev_size=50, test_size=0)

# Tell DSPy that the 'question' field is the input. Any other fields are labels and/or metadata.
trainset = [x.with_inputs('question') for x in dataset.train]
devset = [x.with_inputs('question') for x in dataset.dev]

len(trainset), len(devset)

################################################
#   Build signature for subtasks of pipeline   #
################################################

class GenerateAnswer(dspy.Signature):
    '''Answer questions with short factoid answers'''
    context = dspy.InputField(desc='may contain relevant facts')
    question = dspy.InputField()
    answer = dspy.OutputField(desc='often between 1 and 5 words')

# build the pipeline using DSPy module
class RAG(dspy.Module):
    # uses CoT and GenerateAnswer submodules 
    def __init__(self, num_passages=3):
        super().__init__()
        self.retrieve = dspy.Retrieve(k=num_passages)
        self.generate_answer = dspy.ChainOfThought(GenerateAnswer)

    # control flow of answering questions 
    def forward(self, question):
        context = self.retrieve(question).passages
        prediction = self.generate_answer(context=context, question=question)
        return dspy.Prediction(context=context, answer=prediction.answer)

#############################
#   Optimize the pipeline   #
#############################

# depends on training set, metric for validation, and teleprompter 
from dspy.teleprompt import BootstrapFewShot 

# Validation logic: check that the predicted answer is correct.
# Also check that the retrieved context does actually contain that answer.
def validate_context_and_answer(example, pred, trace=None):
    answer_EM = dspy.evaluate.answer_exact_match(example, pred)
    answer_PM = dspy.evaluate.answer_passage_match(example, pred)
    return answer_EM and answer_PM

# Set up a basic teleprompter, which will compile our RAG program.
teleprompter = BootstrapFewShot(metric=validate_context_and_answer)
compiled_rag = teleprompter.compile(RAG(), trainset=trainset)

##############################
#   Executing the pipeline   #
##############################

# Ask any question you like to this simple RAG program.
my_question = "What castle did David Gregory inherit?"
# Get the prediction. This contains `pred.context` and `pred.answer`.
pred = compiled_rag(my_question)
print(f"Question: {my_question}")
print(f"Predicted Answer: {pred.answer}")
print(f'Retrieved Contexts (truncated): {[c[:200] + '...' for c in pred.context]}')

turbo.inspect_history(n=1) # inspect last prompt 

# Even though we haven't written any of this detailed demonstrations, we see that DSPy was able to
# bootstrap this 3,000 token prompt for 3-shot retrieval-augmented generation with hard negative
# passages and uses Chain-of-Thought reasoning within an extremely simply-written program.

#######################################################################
#   Evaluating the Pipeline (the accuracy or exact match of answer)   #
#######################################################################

from dspy.evaluate.evaluate import Evaluate 

# Set up the `evaluate_on_hotpotqa` function. We'll use this many times below.
evaluate_on_hotpotqa = Evaluate(devset=devset, num_threads=1, display_progress=False, display_table=5)

# Evaluate the `compiled_rag` program with the `answer_exact_match` metric.
metric = dspy.evaluate.answer_exact_match 
evaluate_on_hotpotqa(compiled_rag, metric=metric)

################################
#   Evaluating the retrieval   #
################################

def gold_passages_retrieved(example, pred, trace=None):
    gold_titles = set(map(dspy.evaluate.normalize_text, example['gold_titles']))
    found_titles = set(map(dspy.evaluate.normalize_text, [c.split(' | ')[0] for c in pred.context]))
    return gold_titles.issubset(found_titles)

compiled_rag_retrival_score = evaluate_on_hotpotqa(compiled_rag, metric=gold_passages_retrieved)
