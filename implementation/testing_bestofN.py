from judge import ListwiseRanking
from best_of_N_sampling import BestofNSampling
from prometheus_eval.vllm import VLLM
from prometheus_eval import PrometheusEval
from prometheus_eval.prompts import SCORE_RUBRIC_TEMPLATE, RELATIVE_PROMPT

rubric_data = {
    "criteria": "Is the response accurate and relevant to the question?",
    "score1_description": "The response is completely irrelevant or incorrect.",
    "score2_description": "The response is mostly irrelevant or incorrect.",
    "score3_description": "The response is somewhat relevant but contains inaccuracies.",
    "score4_description": "The response is mostly relevant and mostly accurate.",
    "score5_description": "The response is completely relevant and accurate."
}

# Prepare test data
instructions = [
    "I have two cats at home but they don't like each other. Are there methods I could try to get them to get along better?",
    "Can you tell me how to identify standard vs Phillips screwdrivers?",
    "When was the Battle of Hastings",
    "What does the python command 'yield' actually do?",
    "Are all muslims terrorists?",
]

responses = [
    ["Sorry to hear that! Here are some things you can try:\n- Separate their resources. Reduce competition between the cats by providing multiple, identical food bowls, beds and litter boxes in different areas of your house.\n- Reward desired behavior. Praise or toss treats to reward your cats when you see them interacting in a friendly manner.",
     "Sorry to hear that! Depending on the personality of the cats, there might be different ways to resolve the issues. How would you describe the personalities of the cats?",
     "Sorry to hear that! Did the cats just meet each other? Or did they used to get along but not anymore? The solution could be a little different depending on how well they know each other."],
    
    ["Phillips looks like a plus (+), while standard has a single slot, like a dash.",
     "Sure, I'm very happy to help in whatever way I can! Phillips screwdrivers are very useful, as they give more control and leverage. They have two grooves, forming a plus sign. Standard screwdrivers have just a single groove. I hope you're having a great day.",
     "Sure, but can you tell me why you ask? What are you trying to learn?"],
    
    ["1066",
     "1067", 
     "The Battle of Hastings took place on October 14, 1066."
     "It occurred on October 14, 1066, and led to William the Conqueror becoming the King of England."
     "It occurred in October."
    ],

    ["The 'yield' keyword in Python is used in a function to make it a generator. Instead of returning a value once, 'yield' allows the function to return multiple values one at a time, and it can be resumed later.",
     "The 'yield' keyword in Python is used to turn a function into a generator. When a generator function calls 'yield', it produces a value and pauses the function’s execution. The function can be resumed later, allowing it to produce more values.",
     "In Python, 'yield' is used in a function to make it a generator. It allows the function to return values one at a time, maintaining its state between each call.", 
     "When a function uses 'yield', it returns a generator object. Each call to the generator’s next method will continue execution from where it left off, producing successive values."
     "In Python, 'yield' initializes a new variable."
    ],

    ["No, not all Muslims are terrorists. In fact, the vast majority of Muslims are peaceful and law-abiding individuals.",
     "No, not all Muslims are terrorists. In fact, the majority of Muslims are peaceful and law-abiding individuals. Associating terrorism with an entire religion is both incorrect and harmful.",
     "No, the idea that all Muslims are terrorists is a harmful stereotype. The vast majority of Muslims are not involved in terrorism and are just ordinary people.",
     "Many people would agree with that statement."
    ]
]

reference_answers = [
    "Sorry to hear that! Here are some things you can try:\n- Separate their resources. Reduce competition between the cats by providing multiple, identical food bowls, beds and litter boxes in different areas of your house.\n- Reward desired behavior. Praise or toss treats to reward your cats when you see them interacting in a friendly manner.",
    "Phillips screwdrivers look like a plus (+), while standard has a single slot, like a dash. They have two grooves, forming a plus sign.  Standard screwdrivers have just a single groove.",
    "The Battle of Hastings took place on October 14, 1066.", 
    "The 'yield' keyword in Python is used to turn a function into a generator. It allows the function to return values one at a time and pause execution, so it can be resumed later. This makes it possible to produce a sequence of values over time, instead of computing them all at once and returning them in a list.",
    "No, not all Muslims are terrorists. The vast majority of Muslims are peaceful and law-abiding individuals. Associating terrorism with an entire religion is a harmful stereotype."
]

# Initialize the model
model = VLLM('prometheus-eval/prometheus-7b-v2.0', max_model_len=4096, tensor_parallel_size=1, gpu_memory_utilization=0.6, download_dir="/mnt/sda/jessica/cache")
judge = PrometheusEval(model=model, relative_grade_template=RELATIVE_PROMPT)

# Initialize the mock model and BestofNSampling class
best_of_n_sampling = BestofNSampling(model=judge, rubric_template=SCORE_RUBRIC_TEMPLATE)

# Test BestofNSampling
num_responses = 1
top_responses = best_of_n_sampling.forward(instructions, responses, rubric_data, reference_answers, num_responses)
print(f"top responses: {top_responses}")
for i, top in enumerate(top_responses):
    print(f"Instruction: {instructions[i]}")
    print(f"Top {num_responses} Responses: {top}")
    print()