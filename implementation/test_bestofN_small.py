import pytest
from prometheus_eval.mock import MockLLM
from best_of_N_sampling import BestofNSampling

@pytest.fixture
def mock_model():
    """
    Fixture to provide a mock model instance for testing.
    """
    return MockLLM(mode="relative")

@pytest.fixture
def best_of_n_sampling(mock_model):
    """
    Fixture to provide a BestofNSampling instance for testing.
    """
    return BestofNSampling(model=mock_model, rubric_template="Some rubric template {some_key}")

@pytest.fixture
def rubric_data():
    """
    Fixture to provide common rubric data for testing.
    """
    return {"some_key": "some_value"}

def test_bestofn_sampling(mock_model, best_of_n_sampling, rubric_data):
    """
    Test case for the BestofNSampling class.
    """
    # Mock DirectAssessment behavior
    def mock_absolute_grade(instructions, responses, rubric, reference_answers):
        # Assuming scores are from 1 to 5
        return ["feedback"] * len(responses), [5, 4, 3]  # Example scores

    # Mock ListwiseRanking behavior
    def mock_listwise_ranking(instructions, response_list, rubric_data, reference_answers):
        # Ranking responses by index, simplest case
        return [[i + 1 for i in range(len(response_list[0]))]]  # Simple rank as [1, 2, 3, ...]

    # Apply mocks to the DirectAssessment and ListwiseRanking methods
    best_of_n_sampling.direct_assessment.forward = mock_absolute_grade
    best_of_n_sampling.listwise_ranking.forward = mock_listwise_ranking

    # Define test data
    instructions = ["instruction1"]
    response_list = [
        ["response1", "response2", "response3"]
    ]
    reference_answers = ["reference1"]

    # Perform the test
    top_n = best_of_n_sampling.forward(
        instructions,
        response_list,
        rubric_data,
        reference_answers,
        num=2  # Number of top responses to select
    )

    # Assertions
    expected_top_n = [["response1", "response2"]]  # Adjust based on expected behavior
    assert top_n == expected_top_n, f"Expected top N responses to be {expected_top_n}, but got {top_n}"

def test_bestofn_sampling_with_more_responses(mock_model, best_of_n_sampling, rubric_data):
    """
    Test case for the BestofNSampling class with more responses than needed.
    """
    # Mock DirectAssessment behavior
    def mock_absolute_grade(instructions, responses, rubric, reference_answers):
        # Assuming scores are from 1 to 5
        return ["feedback"] * len(responses), [5, 4, 3, 2, 1]  # Example scores

    # Mock ListwiseRanking behavior
    def mock_listwise_ranking(instructions, response_list, rubric_data, reference_answers):
        # Ranking responses by index, simplest case
        return [[i + 1 for i in range(len(response_list[0]))]]  # Simple rank as [1, 2, 3, ...]

    # Apply mocks to the DirectAssessment and ListwiseRanking methods
    best_of_n_sampling.direct_assessment.forward = mock_absolute_grade
    best_of_n_sampling.listwise_ranking.forward = mock_listwise_ranking

    # Define test data
    instructions = ["instruction1"]
    response_list = [
        ["response1", "response2", "response3", "response4", "response5"]
    ]
    reference_answers = ["reference1"]

    # Perform the test
    top_n = best_of_n_sampling.forward(
        instructions,
        response_list,
        rubric_data,
        reference_answers,
        num=3  # Number of top responses to select
    )

    # Assertions
    expected_top_n = [["response1", "response2", "response3"]]  # Adjust based on expected behavior
    assert top_n == expected_top_n, f"Expected top N responses to be {expected_top_n}, but got {top_n}"
