from data_filter import ResponseFilter, DifficultyFilter
from prometheus_eval.vllm import VLLM
from prometheus_eval import PrometheusEval
from prometheus_eval.prompts import SCORE_RUBRIC_TEMPLATE, RELATIVE_PROMPT

# Prepare test data for response filter 
instructions_rf = [
    "I have two cats at home but they don't like each other. Are there methods I could try to get them to get along better?",
    "Can you tell me how to identify standard vs Phillips screwdrivers?",
    "When was the Battle of Hastings",
    "What does the python command 'yield' actually do?",
    "Are all muslims terrorists?",
]

responses_rf = [
    ["Sorry to hear that! Here are some things you can try:\n- Separate their resources. Reduce competition between the cats by providing multiple, identical food bowls, beds and litter boxes in different areas of your house.\n- Reward desired behavior. Praise or toss treats to reward your cats when you see them interacting in a friendly manner.",
     "Sorry to hear that! Depending on the personality of the cats, there might be different ways to resolve the issues. How would you describe the personalities of the cats?",
     "Sorry to hear that! Did the cats just meet each other? Or did they used to get along but not anymore? The solution could be a little different depending on how well they know each other."],
    
    ["Phillips looks like a plus (+), while standard has a single slot, like a dash.",
     "Sure, I'm very happy to help in whatever way I can! Phillips screwdrivers are very useful, as they give more control and leverage. They have two grooves, forming a plus sign. Standard screwdrivers have just a single groove. I hope you're having a great day.",
     "Sure, but can you tell me why you ask? What are you trying to learn?"],
    
    ["1066",
     "1067", 
     "The Battle of Hastings took place on October 14, 1066.",
     "It occurred on October 14, 1066, and led to William the Conqueror becoming the King of England.",
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

# Initialize the model
model = VLLM('prometheus-eval/prometheus-7b-v2.0', max_model_len=4096, tensor_parallel_size=1, gpu_memory_utilization=0.6, download_dir="/mnt/sda/jessica/cache")
judge = PrometheusEval(model=model, relative_grade_template=RELATIVE_PROMPT)

response_filter = ResponseFilter(judge)
best_responses_rf = response_filter.forward(instructions_rf, responses_rf)
print(f"Best responses (Response Filter): {best_responses_rf}")


# Prepare test data for difficulty filter 
metaprompt_df = "Generate a challenging math problem" # SINGLE STRING??? 
instructions_df = [
    "Calculate the integral of \( \frac{e^{x^2}}{x} \) from 1 to infinity.",
    "Determine the eigenvalues of the matrix \( \begin{pmatrix} 3 & 2 & -1 \\ 2 & -3 & 4 \\ -1 & 4 & -2 \end{pmatrix} \).",
    "Solve for \( x \) in the equation \( \ln(x^2 + 3x - 4) = \cos^{-1}(2x - 1) \).",
    "Find the Taylor series expansion of \( \sin(x^3) \) up to the \( x^7 \) term.",
    "Determine the sum of the infinite series \( \sum_{n=1}^{\infty} \frac{(-1)^{n+1} n!}{(2n)!} \)."
]

difficulty_filter = DifficultyFilter(judge)
best_responses_df = difficulty_filter.forward(metaprompt_df, instructions_df)
print(f"Best responses (Difficulty Filter): {best_responses_df}")

