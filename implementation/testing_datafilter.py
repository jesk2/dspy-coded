import unittest
from DataFilter import ResponseFilter, DifficultyFilter
from mock import MockModel

class TestFilters(unittest.TestCase):
    def setUp(self):
        self.model = MockModel()  # Placeholder for an actual model
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

    def test_difficulty_filter(self):
        instructions = [
            "Generate a basic recipe for a quick breakfast dish.",
            "Write a detailed guide on designing a workout routine for beginners.",
            "Draft a comprehensive project proposal for a new software application.",
            "Compose a thorough legal contract for a freelance services agreement.",
            "Design a training module outline for onboarding new employees in a tech company.",
            "Generate a detailed problem set on advanced topics in quantum mechanics."
        ]
        
        responses = [
            [
                "Mix flour, sugar, and eggs, and bake for 20 minutes.",
                "Prepare a simple fruit smoothie with bananas and strawberries."
            ],  # Responses for basic recipe
            [
                "Include warm-up exercises, strength training, and cool-down stretches.",
                "Design a 30-minute workout with basic cardio and bodyweight exercises."
            ],  # Responses for workout routine
            [
                "Outline goals, timelines, resources, and deliverables.",
                "Create a detailed project plan including milestones and budget estimates."
            ],  # Responses for project proposal
            [
                "Specify terms of the rental, including duration, deposit, and conditions.",
                "Draft a legal document outlining service agreements and client obligations."
            ],  # Responses for legal contract
            [
                "Include modules on company policies, job roles, and safety procedures.",
                "Develop a training module with interactive lessons and assessments."
            ],  # Responses for training module
            [
                "Calculate the probability of particle decay in a quantum field.",
                "Create a set of problems on quantum entanglement and wave functions."
            ]  # Responses for quantum mechanics problem
        ]
        
        reference_answers = [None] * len(instructions)
        
        # Apply the DifficultyFilter
        sorted_instructions = self.difficulty_filter.forward(instructions)

        self.assertIsInstance(sorted_instructions, list)
        self.assertTrue(all(isinstance(instr_list, list) for instr_list in sorted_instructions))
        self.assertEqual(len(sorted_instructions), len(instructions))
        
        # Check that the sorting is based on difficulty
        # The most complex task should be first, followed by next complex, etc.
        self.assertEqual(sorted_instructions[0][0], "Generate a detailed problem set on advanced topics in quantum mechanics.")
        self.assertEqual(sorted_instructions[1][0], "Draft a comprehensive project proposal for a new software application.")
        self.assertEqual(sorted_instructions[2][0], "Compose a thorough legal contract for a freelance services agreement.")
        self.assertEqual(sorted_instructions[3][0], "Design a training module outline for onboarding new employees in a tech company.")
        self.assertEqual(sorted_instructions[4][0], "Write a detailed guide on designing a workout routine for beginners.")
        self.assertEqual(sorted_instructions[5][0], "Generate a basic recipe for a quick breakfast dish.")
    
    def test_detailed_difficulty_filter(self):
        instructions = [
            "Explain the significance of the Treaty of Versailles in World War II.",
            "Solve the integral of e^x * sin(x) dx.",
            "Discuss the ethical implications of artificial intelligence in healthcare."
        ]
        feedbacks, scores = self.difficulty_filter.forward(instructions)
        self.assertTrue(all(1 <= score <= 5 for score in scores))


if __name__ == '__main__':
    unittest.main()
