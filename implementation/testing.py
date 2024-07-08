from judge import DirectAssessment, PairwiseRanking, ListwiseRanking
from DataFilter import ResponseFilter, DifficultyFilter, DiversityFilter
from prometheus_eval.prompts import ABSOLUTE_PROMPT, SCORE_RUBRIC_TEMPLATE, RELATIVE_PROMPT


# Define the rubric data
rubric_data = {
    "criteria": "Is the model proficient in applying empathy and emotional intelligence to its responses when the user conveys emotions or faces challenging circumstances?",
    "score1_description": "The model neglects to identify or react to the emotional tone of user inputs, giving responses that are unfitting or emotionally insensitive.",
    "score2_description": "The model intermittently acknowledges emotional context but often responds without sufficient empathy or emotional understanding.",
    "score3_description": "The model typically identifies emotional context and attempts to answer with empathy, yet the responses might sometimes miss the point or lack emotional profundity.",
    "score4_description": "The model consistently identifies and reacts suitably to emotional context, providing empathetic responses. Nonetheless, there may still be sporadic oversights or deficiencies in emotional depth.",
    "score5_description": "The model excels in identifying emotional context and persistently offers empathetic, emotionally aware responses that demonstrate a profound comprehension of the user's emotions or situation."
}

# Prepare test data
instructions = [
    "How can I improve my mental health?",
    "What should I do if I'm feeling very anxious?",
    "How do I support a friend going through a tough time?"
]
responses = [
    ["You should talk to a therapist and exercise regularly.", "Just take deep breaths and relax.", "Be there for them and listen to what they have to say."],
    ["It's important to seek professional help and maintain a healthy lifestyle.", "Find activities that calm you and do them regularly.", "Offer your support and let them know you're there for them."],
    ["Consider therapy and regular physical activity to boost mental health.", "Anxiety can be managed with relaxation techniques and a support network.", "Show empathy and be a good listener."]
]
reference_answers = [
    "Talking to a therapist and exercising regularly are good ways to improve mental health.",
    "Taking deep breaths and finding ways to relax can help with anxiety.",
    "Supporting a friend involves being there for them and listening to them."
]

# Initialize the filters
model_name = "prometheus-eval/prometheus-7b-v2.0"
rubric_template = SCORE_RUBRIC_TEMPLATE

response_filter = ResponseFilter(model_name, rubric_template)
difficulty_filter = DifficultyFilter(model_name, rubric_template)
diversity_filter = DiversityFilter(model_name, rubric_template)

# Run the filters
filtered_responses = response_filter.forward(instructions, responses, reference_answers, rubric_data)
challenging_instructions = difficulty_filter.forward(instructions, reference_answers, rubric_data)
diverse_instructions = diversity_filter.forward(instructions, rubric_data)

# Print the results
print("Filtered Responses:", filtered_responses)
print("Challenging Instructions:", challenging_instructions)
print("Diverse Instructions:", diverse_instructions)