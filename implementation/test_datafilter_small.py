import pytest
from prometheus_eval.mock import MockLLM
from data_filter import QualityResponseFilter, DifficultyFilter, ResponseData, InstructionData

@pytest.fixture
def mock_model():
    """
    Fixture to provide a mock model instance for testing.
    """
    return MockLLM(mode="absolute")

@pytest.fixture
def rubric_data():
    """
    Fixture to provide common rubric data for testing.
    """
    return {
        "criteria": "Evaluate the quality of the response on these three metrics: clear and accurate reply, relevant and thorough information, and polite tone.",
        "score1_description": "The response is unclear in providing an accurate reply to the instruction, is incomplete, and has an impolite tone.",
        "score2_description": "The response provides a somewhat clear and accurate reply, but is incomplete, has irrelevant information, and/or is impolite.",
        "score3_description": "The response is helpful and accurate, somewhat detailed and relevant to the instruction, with a neutral or polite tone.",
        "score4_description": "The response is overall helpful, clear and accurate. It includes many relevant details in a thorough way, and uses a polite tone.",
        "score5_description": "The response is extremely clear and helpful, with very accurate information. It includes all relevant details in a thorough way, and uses a polite tone."
    }

def test_response_filter(mock_model, rubric_data):
    filter = QualityResponseFilter(model=mock_model, score_threshold=4)

    instructions = ["instruction1", "instruction2"]
    responses = [["response1", "response2"], ["response3", "response4"]]

    # Mock the behavior of the model
    mock_model.absolute_grade = lambda instructions, responses, rubric, reference_answers: (
        [["feedback1", "feedback2"], ["feedback3", "feedback4"]],
        [[4, 5]]  
    )
    
    # Perform the test
    quality_responses = filter.forward(instructions, responses)
    
    # Create expected ResponseData objects
    expected = [
        ResponseData(instruction="instruction1", response="response1", score=4.0),
        ResponseData(instruction="instruction1", response="response2", score=5.0),
        ResponseData(instruction="instruction2", response="response3", score=4.0),
        ResponseData(instruction="instruction2", response="response4", score=5.0)
    ]
    
    assert quality_responses == expected

def test_difficulty_filter(mock_model, rubric_data):
    """
    Test case for the DifficultyFilter class.
    """
    filter = DifficultyFilter(model=mock_model, score_threshold=4)

    metaprompt = "Assess the difficulty of the instructions."
    instructions = ["instruction1", "instruction2"]

    # Mock the behavior of the model
    mock_model.absolute_grade = lambda instructions, responses, rubric, reference_answers: (
        [["feedback1"], ["feedback2"]],
        [[4]]
    )
    
    # Perform the test
    difficult_instructions = filter.forward(metaprompt, instructions)
    
    expected = [
        InstructionData(instruction="instruction1", response="instruction1", score=4.0),
        InstructionData(instruction="instruction2", response="instruction2", score=4.0)
    ]
    
    assert difficult_instructions == expected
