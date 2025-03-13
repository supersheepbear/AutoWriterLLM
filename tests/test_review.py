import pytest
from unittest.mock import MagicMock, patch
from src.autowriterllm.review import ContentReviewer, ReviewFeedback
from src.autowriterllm.exceptions import ReviewError

@pytest.fixture
def mock_ai_client():
    """Mock AI client for testing."""
    client = MagicMock()
    response = MagicMock()
    response.content = [MagicMock(text="""
technical_accuracy: 0.8
clarity: 0.7
completeness: 0.9
comments:
  - "Good explanation of concepts"
  - "Could use more examples"
suggested_improvements:
  - "Add more practical examples"
  - "Include diagrams for visual learners"
    """)]
    client.messages.create.return_value = response
    return client

def test_review_content_with_content_type(mock_ai_client):
    """Test reviewing content with different content types."""
    reviewer = ContentReviewer(mock_ai_client)
    content = "This is test content for review."
    section_title = "Test Section"
    
    # Test with programming content type
    review = reviewer.review_content(content, section_title, "programming")
    
    # Verify the review was created correctly
    assert review.section_title == section_title
    assert review.technical_accuracy == 0.8
    assert review.clarity == 0.7
    assert review.completeness == 0.9
    assert len(review.comments) == 2
    assert len(review.suggested_improvements) == 2
    
    # Verify the AI client was called with the correct prompt
    mock_ai_client.messages.create.assert_called_once()
    call_kwargs = mock_ai_client.messages.create.call_args[1]
    assert "messages" in call_kwargs
    
    # Reset mock for next test
    mock_ai_client.reset_mock()
    
    # Test with science content type
    review = reviewer.review_content(content, section_title, "science")
    
    # Verify the AI client was called with a different prompt
    mock_ai_client.messages.create.assert_called_once()
    call_kwargs = mock_ai_client.messages.create.call_args[1]
    assert "messages" in call_kwargs
    
    # The prompt should contain science-specific instructions
    messages = call_kwargs.get("messages", [])
    assert len(messages) > 0
    prompt = messages[0].get("content", "")
    assert "science" in prompt.lower()

def test_review_error_handling(mock_ai_client):
    """Test error handling in the review process."""
    reviewer = ContentReviewer(mock_ai_client)
    
    # Make the AI client raise an exception
    mock_ai_client.messages.create.side_effect = Exception("API error")
    
    # Verify that the error is properly caught and wrapped
    with pytest.raises(ReviewError, match="Review failed: API error"):
        reviewer.review_content("content", "section", "generic")

def test_review_missing_fields(mock_ai_client):
    """Test handling of missing fields in review response."""
    # Create a response with missing fields
    response = MagicMock()
    response.content = [MagicMock(text="""
technical_accuracy: 0.8
clarity: 0.7
# completeness is missing
comments:
  - "Good explanation of concepts"
suggested_improvements:
  - "Add more examples"
    """)]
    mock_ai_client.messages.create.return_value = response
    
    reviewer = ContentReviewer(mock_ai_client)
    
    # Verify that missing fields trigger an error
    with pytest.raises(ReviewError, match="Missing required field in review: completeness"):
        reviewer.review_content("content", "section", "generic")

def test_create_review_prompt_for_different_content_types():
    """Test that different content types generate different prompts."""
    reviewer = ContentReviewer(MagicMock())
    content = "Test content"
    section = "Test section"
    
    # Generate prompts for different content types
    programming_prompt = reviewer._create_review_prompt(content, section, "programming")
    science_prompt = reviewer._create_review_prompt(content, section, "science")
    history_prompt = reviewer._create_review_prompt(content, section, "history")
    generic_prompt = reviewer._create_review_prompt(content, section, "generic")
    
    # Verify that each prompt contains content-specific instructions
    assert "programming" in programming_prompt.lower()
    assert "code" in programming_prompt.lower()
    
    assert "science" in science_prompt.lower()
    assert "scientific" in science_prompt.lower()
    
    assert "history" in history_prompt.lower()
    assert "historical" in history_prompt.lower()
    
    # Verify that the generic prompt doesn't contain specific instructions
    assert "programming" not in generic_prompt.lower()
    assert "science" not in generic_prompt.lower()
    assert "history" not in generic_prompt.lower() 