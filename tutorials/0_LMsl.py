# GIVES ERROR AS OF NOW !!!

from openai import OpenAI
import dspy

gpt3_turbo = dspy.OpenAI(model='gpt-3.5-turbo-1106', max_tokens=300)
dspy.configure(lm=gpt3_turbo)

# directly call LLM by giving it a raw prompt 
gpt3_turbo("hello! this is a raw prompt to GPT-3.5")


