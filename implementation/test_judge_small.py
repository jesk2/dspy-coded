import pytest
from prometheus_eval.mock import MockLLM
from judge import DirectAssessment, PairwiseRanking, ListwiseRanking  

@pytest.fixture
def mock_model():
    """
    Fixture to provide a mock model instance for testing.
    """
    return MockLLM()

@pytest.fixture
def rubric_data():
    """
    Fixture to provide common rubric data for testing.
    """
    return {"some_key": "some_value"}

def test_direct_assessment(mock_model, rubric_data):
    """
    Test case for the DirectAssessment class.
    """
    # Setup DirectAssessment with the mock model and a template
    assessment = DirectAssessment(mock_model, "Some rubric template {some_key}")

    # Define test data
    instructions = ["instruction1", "instruction2"]
    responses = ["response1", "response2"]
    reference_answers = ["reference1", "reference2"]

    # Mock the behavior of the model
    mock_model.absolute_grade = lambda instructions, responses, rubric, reference_answers: (
        ["Hello [RESULT] 5"] * len(responses), [1.0] * len(responses)
    )

    # Perform the test
    feedbacks, scores = assessment.forward(instructions, responses, rubric_data, reference_answers)
    
    assert feedbacks == ["Hello [RESULT] 5", "Hello [RESULT] 5"]
    assert scores == [1.0, 1.0]

def test_pairwise_ranking(mock_model, rubric_data):
    """
    Test case for the PairwiseRanking class.
    """
    # Setup PairwiseRanking with the mock model and a template
    ranking = PairwiseRanking(mock_model, "Some rubric template {some_key}")

    # Define test data
    instructions = ["instruction1"]
    responseA = ["responseA1"]
    responseB = ["responseB1"]
    reference_answers = ["reference1"]

    # Mock the behavior of the model
    mock_model.relative_grade = lambda instructions, responses_A, responses_B, rubric, reference_answers: (
        ["feedback1"], ["A"]
    )
    
    # Perform the test
    feedbacks, winners = ranking.forward(instructions, responseA, responseB, rubric_data, reference_answers)
    
    assert feedbacks == ["feedback1"]
    assert winners == ["A"]

def test_listwise_ranking(mock_model, rubric_data):
    """
    Test case for the ListwiseRanking class.
    """
    # Setup ListwiseRanking with the mock model and a template
    ranking = ListwiseRanking(mock_model, "Some rubric template {some_key}")

    # Define test data
    instructions = ["instruction1"]
    response_list = [["response1", "response2"]]
    reference_answers = ["reference1"]

    # Mock the behavior of the model
    # Assuming mock behavior returns feedback and winner as 'A' if the first response is better
    mock_model.relative_grade = lambda instructions, responses_A, responses_B, rubric, reference_answers: (
        ["Hello [RESULT] A"], ["A"] if responses_A[0] == "response1" else ["B"]
    )

    # Perform the test
    rankings = ranking.forward(instructions, response_list, rubric_data, reference_answers)

    # Assuming the ranking logic will assign rank based on the mock result
    assert rankings == [[1, 2]]  # Adjust based on actual ranking logic