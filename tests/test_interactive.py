import pytest
from unittest.mock import MagicMock, patch
import time
from src.autowriterllm.interactive import InteractiveElementsGenerator, QuizQuestion, Exercise
from src.autowriterllm.exceptions import GenerationError

@pytest.fixture
def mock_ai_client():
    """Mock AI client for testing."""
    client = MagicMock()
    response = MagicMock()
    response.content = [MagicMock(text="""
questions:
  - question: "What is the main concept?"
    options:
      - "Option A"
      - "Option B"
      - "Option C"
    correct_answer: "Option B"
    explanation: "Explanation for the answer"
    type: "multiple_choice"
    """)]
    client.messages.create.return_value = response
    return client

def test_generate_quiz_with_content_type(mock_ai_client):
    """Test generating quiz questions with different content types."""
    generator = InteractiveElementsGenerator(mock_ai_client)
    content = "This is test content for quiz generation."
    
    # Test with programming content type
    generator.generate_quiz(content, "medium", "programming")
    
    # Verify the AI client was called with the correct prompt
    mock_ai_client.messages.create.assert_called_once()
    call_kwargs = mock_ai_client.messages.create.call_args[1]
    assert "messages" in call_kwargs
    
    # The prompt should contain programming-specific instructions
    messages = call_kwargs.get("messages", [])
    assert len(messages) > 0
    prompt = messages[0].get("content", "")
    assert "programming" in prompt.lower()
    
    # Reset mock for next test
    mock_ai_client.reset_mock()
    
    # Test with science content type
    generator.generate_quiz(content, "medium", "science")
    
    # Verify the AI client was called with a different prompt
    mock_ai_client.messages.create.assert_called_once()
    call_kwargs = mock_ai_client.messages.create.call_args[1]
    assert "messages" in call_kwargs
    
    # The prompt should contain science-specific instructions
    messages = call_kwargs.get("messages", [])
    assert len(messages) > 0
    prompt = messages[0].get("content", "")
    assert "science" in prompt.lower()

def test_generate_exercises_with_content_type(mock_ai_client):
    """Test generating exercises with different content types."""
    generator = InteractiveElementsGenerator(mock_ai_client)
    content = "This is test content for exercise generation."
    
    # Test with programming content type
    generator.generate_exercises(content, "intermediate", "programming")
    
    # Verify the AI client was called with the correct prompt
    mock_ai_client.messages.create.assert_called_once()
    call_kwargs = mock_ai_client.messages.create.call_args[1]
    assert "messages" in call_kwargs
    
    # The prompt should contain programming-specific instructions
    messages = call_kwargs.get("messages", [])
    assert len(messages) > 0
    prompt = messages[0].get("content", "")
    assert "programming" in prompt.lower()
    
    # Reset mock for next test
    mock_ai_client.reset_mock()
    
    # Test with mathematics content type
    generator.generate_exercises(content, "intermediate", "mathematics")
    
    # Verify the AI client was called with a different prompt
    mock_ai_client.messages.create.assert_called_once()
    call_kwargs = mock_ai_client.messages.create.call_args[1]
    assert "messages" in call_kwargs
    
    # The prompt should contain mathematics-specific instructions
    messages = call_kwargs.get("messages", [])
    assert len(messages) > 0
    prompt = messages[0].get("content", "")
    assert "mathematics" in prompt.lower()

def test_error_handling_in_quiz_generation(mock_ai_client):
    """Test error handling in quiz generation."""
    generator = InteractiveElementsGenerator(mock_ai_client)
    
    # Make the AI client raise an exception
    mock_ai_client.messages.create.side_effect = Exception("API error")
    
    # Verify that the error is properly caught and wrapped
    with pytest.raises(GenerationError, match="Quiz generation failed: API error"):
        generator.generate_quiz("content", "medium", "generic")

def test_error_handling_in_exercise_generation(mock_ai_client):
    """Test error handling in exercise generation."""
    generator = InteractiveElementsGenerator(mock_ai_client)
    
    # Make the AI client raise an exception
    mock_ai_client.messages.create.side_effect = Exception("API error")
    
    # Verify that the error is properly caught and wrapped
    with pytest.raises(GenerationError, match="Exercise generation failed: API error"):
        generator.generate_exercises("content", "intermediate", "generic")

def test_retry_logic(mock_ai_client):
    """Test retry logic in content generation."""
    generator = InteractiveElementsGenerator(mock_ai_client)
    
    # Make the first call fail, then succeed
    mock_ai_client.messages.create.side_effect = [Exception("Temporary error"), mock_ai_client.messages.create.return_value]
    
    # Patch time.sleep to avoid waiting in tests
    with patch("time.sleep"):
        # This should succeed after one retry
        generator.generate_quiz("content", "medium", "generic")
    
    # Verify that the client was called twice
    assert mock_ai_client.messages.create.call_count == 2

def test_create_quiz_prompt_for_different_content_types():
    """Test that different content types generate different quiz prompts."""
    generator = InteractiveElementsGenerator(MagicMock())
    content = "Test content"
    difficulty = "medium"
    
    # Generate prompts for different content types
    programming_prompt = generator._create_quiz_prompt(content, difficulty, "programming")
    science_prompt = generator._create_quiz_prompt(content, difficulty, "science")
    history_prompt = generator._create_quiz_prompt(content, difficulty, "history")
    generic_prompt = generator._create_quiz_prompt(content, difficulty, "generic")
    
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

def test_create_exercise_prompt_for_different_content_types():
    """Test that different content types generate different exercise prompts."""
    generator = InteractiveElementsGenerator(MagicMock())
    content = "Test content"
    level = "intermediate"
    
    # Generate prompts for different content types
    programming_prompt = generator._create_exercise_prompt(content, level, "programming")
    science_prompt = generator._create_exercise_prompt(content, level, "science")
    history_prompt = generator._create_exercise_prompt(content, level, "history")
    generic_prompt = generator._create_exercise_prompt(content, level, "generic")
    
    # Verify that each prompt contains content-specific instructions
    assert "programming" in programming_prompt.lower()
    assert "code" in programming_prompt.lower()
    
    assert "science" in science_prompt.lower()
    assert "scientific" in science_prompt.lower() or "experiment" in science_prompt.lower()
    
    assert "history" in history_prompt.lower()
    assert "historical" in history_prompt.lower()
    
    # Verify that the generic prompt doesn't contain specific instructions
    assert "programming" not in generic_prompt.lower()
    assert "science" not in generic_prompt.lower()
    assert "history" not in generic_prompt.lower() 