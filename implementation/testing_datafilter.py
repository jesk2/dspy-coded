import unittest
from DataFilter import ResponseFilter, DifficultyFilter
from mock import MockModel

class TestFilters(unittest.TestCase):
    def setUp(self):
        self.model = MockModel()  
        self.response_filter = ResponseFilter(self.model)
        self.difficulty_filter = DifficultyFilter(self.model)

    def test_response_filter(self):
        instructions = [
            "What is the capital of France?",
            "Explain the process of photosynthesis.",
            "What are the benefits of regular exercise?"
        ]
        responses = [
            ["The capital of France is Paris.", "Paris.", "The capital is Paris."],
            ["Photosynthesis is the process by which green plants use sunlight.", "Plants use sunlight to make food.", "Photosynthesis converts light energy into chemical energy."],
            ["Regular exercise helps maintain health.", "Exercise is good for health.", "Exercise provides numerous health benefits."]
        ]
        reference_answers = [
            "Paris is the capital of France.",
            "Photosynthesis involves the conversion of light energy into chemical energy by green plants.",
            "Regular exercise provides numerous health benefits including weight management and disease prevention."
        ]
        best_responses = self.response_filter.forward(instructions, responses, reference_answers)
        self.assertTrue(len(best_responses) <= 5)

    def test_detailed_response_filter(self):
        instructions = [
            "Describe the impact of climate change on polar bear populations.",
            "Outline the main events of the French Revolution.",
            "How does quantum computing differ from classical computing?"
        ]
        responses = [
            ["Climate change is causing the ice caps to melt.", "Ice melting affects polar bears.", "Climate change impacts polar bears by reducing their habitat."],
            ["The French Revolution was a period of radical change.", "Main events include the storming of the Bastille.", "The French Revolution saw many significant events such as the Reign of Terror."],
            ["Quantum computing uses qubits.", "Qubits are used in quantum computing.", "Quantum computing is different from classical computing."]
        ]
        reference_answers = [
            "Climate change leads to melting ice caps, loss of habitat, and decreased food availability, negatively impacting polar bears.",
            "Key events of the French Revolution include the storming of the Bastille, the Reign of Terror, and the rise of Napoleon Bonaparte.",
            "Quantum computing leverages quantum mechanics to perform complex calculations more efficiently than classical computers."
        ]
        best_responses = self.response_filter.forward(instructions, responses, reference_answers)
        self.assertTrue(len(best_responses) <= 5)


if __name__ == '__main__':
    unittest.main()
