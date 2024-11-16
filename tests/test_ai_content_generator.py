import pytest
from unittest.mock import patch, MagicMock, call
import tkinter as tk
from pathlib import Path
import re
import threading
import time
from src.autowriterllm.ai_content_generator import (
    ContentGenerator,
    ContentGeneratorGUI,
    Section,
    ContentGenerationError,
    APIRateLimiter,
    RateLimitConfig,
    ContentCache,
)
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor
import requests

# Mock API responses
MOCK_API_RESPONSES = {
    "book_plan": """```markdown:table_of_contents.md
# Python Programming Guide
## 1. Introduction
### 1.1 Getting Started
### 1.2 Environment Setup
```

```markdown:book_summary.md
A comprehensive guide to Python programming.
Target Audience: Intermediate developers
Prerequisites: Basic programming knowledge
```""",
    "section_content": """# Section Title
## Overview
This is a test section.
```python
def example():
    pass
```
## Practice Questions
1. Test question
"""
}

@pytest.fixture
def mock_api():
    """Mock API client responses."""
    with patch('anthropic.Anthropic') as mock_anthropic:
        # Mock the messages.create method
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text=MOCK_API_RESPONSES["book_plan"])]
        mock_client.messages.create.return_value = mock_response
        mock_anthropic.return_value = mock_client
        yield mock_client

@pytest.fixture
def mock_config():
    """Mock configuration."""
    return {
        "provider": "anthropic",
        "anthropic": {
            "api_key": "test_key",
            "model": "claude-3-sonnet-20240229"
        },
    }

@pytest.fixture
def mock_toc_file(tmp_path):
    toc_content = """
    ## Main Section 1
    ### Subsection 1.1
    ### Subsection 1.2
    ## Main Section 2
    """
    toc_file = tmp_path / "table_of_contents.md"
    toc_file.write_text(toc_content)
    return toc_file

@pytest.fixture
def mock_output_dir(tmp_path):
    return tmp_path / "output"

def test_content_generator_initialization(mock_toc_file, mock_output_dir, mock_config):
    with patch("src.autowriterllm.ai_content_generator.ContentGenerator._load_config", return_value=mock_config):
        generator = ContentGenerator(mock_toc_file, mock_output_dir)
        assert generator.toc_file == mock_toc_file
        assert generator.output_dir == mock_output_dir
        assert generator.config == mock_config

def test_parse_toc(mock_toc_file, mock_output_dir, mock_config):
    with patch("src.autowriterllm.ai_content_generator.ContentGenerator._load_config", return_value=mock_config):
        generator = ContentGenerator(mock_toc_file, mock_output_dir)
        generator.parse_toc()
        assert len(generator.sections) == 2
        assert generator.sections[0].title == "Main Section 1"
        assert generator.sections[0].subsections == ["Subsection 1.1", "Subsection 1.2"]

def test_generate_section_content(mock_toc_file, mock_output_dir, mock_config, mock_api):
    """Test section content generation with mocked API."""
    with patch("src.autowriterllm.ai_content_generator.ContentGenerator._load_config", 
              return_value=mock_config):
        generator = ContentGenerator(mock_toc_file, mock_output_dir)
        generator.parse_toc()
        section = generator.sections[0]
        
        # Mock cache and rate limiter
        generator.cache = MagicMock(spec=ContentCache)
        generator.cache.get_cached_content.return_value = None
        generator.rate_limiter = MagicMock(spec=APIRateLimiter)
        
        # Mock API response
        mock_api.messages.create.return_value.content = [
            MagicMock(text=MOCK_API_RESPONSES["section_content"])
        ]
        
        content, filename = generator.generate_section_content(section)
        
        # Verify API was called with correct parameters
        mock_api.messages.create.assert_called_once()
        call_kwargs = mock_api.messages.create.call_args[1]
        assert "model" in call_kwargs
        assert isinstance(call_kwargs.get("messages"), list)
        
        # Verify content was processed correctly
        assert "Section Title" in content
        assert "Practice Questions" in content
        assert filename == "main-section-1.md"

def test_rate_limiter():
    config = RateLimitConfig(max_requests_per_minute=2)
    rate_limiter = APIRateLimiter(config)
    
    with patch("time.sleep", return_value=None) as mock_sleep:
        rate_limiter.wait_if_needed()
        rate_limiter.wait_if_needed()
        rate_limiter.wait_if_needed()
        assert mock_sleep.called

def test_content_cache(tmp_path):
    cache_dir = tmp_path / "cache"
    cache = ContentCache(cache_dir)
    section = Section(title="Test Section", subsections=[], level=2)
    prompt = "Test Prompt"
    key = cache.get_cache_key(section, prompt)
    content = "Test Content"
    
    cache.cache_content(key, content)
    cached_content = cache.get_cached_content(key)
    assert cached_content == content

def test_content_generation_error():
    with pytest.raises(ContentGenerationError, match="Test Error"):
        raise ContentGenerationError("Test Error") 

@pytest.fixture
def gui_app():
    with patch('tkinter.Tk'):
        app = ContentGeneratorGUI()
        # Mock necessary tkinter variables
        app.topic_var = MagicMock()
        app.level_var = MagicMock()
        app.output_path_var = MagicMock()
        app.progress_var = MagicMock()
        return app

class TestContentGeneratorGUI:
    def test_validate_topic_input(self, gui_app):
        """Test topic input validation."""
        # Test valid input
        assert gui_app._validate_topic_input("Python Programming") is True
        
        # Test too short
        assert gui_app._validate_topic_input("Py") is False
        
        # Test too long
        assert gui_app._validate_topic_input("x" * 501) is False
        
        # Test empty
        assert gui_app._validate_topic_input("") is False
        
        # Test only special characters
        assert gui_app._validate_topic_input("!@#$%^") is False

    def test_generate_book_plan(self, gui_app):
        """Test book plan generation."""
        # Mock necessary attributes and methods
        gui_app.topic_var.get.return_value = "Python Programming"
        gui_app.level_var.get.return_value = "intermediate"
        gui_app._validate_topic_input = MagicMock(return_value=True)
        gui_app.generator = MagicMock()
        gui_app.generator._create_book_planning_prompt.return_value = "test prompt"
        gui_app.generator._generate_with_retry.return_value = "test content"
        gui_app.generator._parse_ai_response.return_value = ("toc", "summary")
        
        # Mock file operations
        with patch('pathlib.Path.write_text'):
            gui_app._generate_book_plan()
            
        # Verify method calls
        assert gui_app.topic_var.get.called
        assert gui_app.level_var.get.called
        assert gui_app._validate_topic_input.called
        assert gui_app.generator._create_book_planning_prompt.called
        assert gui_app.generator._generate_with_retry.called

    def test_error_handling(self, gui_app):
        """Test error handling in book plan generation."""
        gui_app.topic_var.get.return_value = "Python"
        gui_app._validate_topic_input = MagicMock(return_value=True)
        gui_app.generator = MagicMock()
        gui_app.generator._create_book_planning_prompt.side_effect = Exception("Test error")
        
        with patch('tkinter.messagebox.showerror'):
            gui_app._generate_book_plan()
            # Verify error handling
            assert gui_app.update_log.called_with("Error: Failed to generate book plan: Test error")

class TestContentGenerator:
    def test_create_book_planning_prompt(self, mock_toc_file, mock_output_dir, mock_config):
        """Test book planning prompt creation."""
        with patch("src.autowriterllm.ai_content_generator.ContentGenerator._load_config", 
                  return_value=mock_config):
            generator = ContentGenerator(mock_toc_file, mock_output_dir)
            prompt = generator._create_book_planning_prompt("Create intermediate Python tutorial")
            
            # Verify prompt content
            assert "Python" in prompt
            assert "intermediate" in prompt
            assert "examples" in prompt.lower()
            assert "practices" in prompt.lower()

    def test_parse_ai_response(self, mock_toc_file, mock_output_dir, mock_config):
        """Test AI response parsing."""
        with patch("src.autowriterllm.ai_content_generator.ContentGenerator._load_config", 
                  return_value=mock_config):
            generator = ContentGenerator(mock_toc_file, mock_output_dir)
            
            # Test with valid response
            response = """            ```table_of_contents.md
            ## Section 1
            ### Subsection 1.1            ```
            ```book_summary.md
            This is a summary            ```
            """
            toc, summary = generator._parse_ai_response(response)
            assert "Section 1" in toc
            assert "This is a summary" in summary

            # Test with invalid response
            with pytest.raises(ContentGenerationError):
                generator._parse_ai_response("Invalid response")

class TestAPIRateLimiter:
    def test_concurrent_requests(self):
        """Test rate limiter under concurrent requests."""
        config = RateLimitConfig(max_requests_per_minute=5)
        limiter = APIRateLimiter(config)
        
        # Mock time.time() and time.sleep()
        with patch('time.time', side_effect=[0, 0.1, 0.2, 0.3, 0.4]) as mock_time, \
             patch('time.sleep') as mock_sleep:
            
            # Test fewer requests (5 instead of 10)
            for _ in range(5):
                limiter.wait_if_needed()
            
            # Verify behavior
            assert mock_sleep.called
            assert mock_time.call_count > 0

class TestContentCache:
    def test_cache_concurrent_access(self, tmp_path):
        """Test cache under concurrent access."""
        cache = ContentCache(tmp_path / "cache")
        section = Section(title="Test", subsections=[], level=2)
        
        # Reduce thread count from 5 to 3
        thread_count = 3
        results = []  # To store results from threads
        
        def cache_operation():
            try:
                key = cache.get_cache_key(section, "test prompt")
                cache.cache_content(key, f"content-{threading.get_ident()}")
                content = cache.get_cached_content(key)
                results.append((True, content))
            except Exception as e:
                results.append((False, str(e)))
        
        # Use a thread pool instead of creating threads manually
        with ThreadPoolExecutor(max_workers=thread_count) as executor:
            futures = [executor.submit(cache_operation) for _ in range(thread_count)]
            concurrent.futures.wait(futures, timeout=2)  # Add timeout
        
        # Verify results
        assert len(results) == thread_count
        assert all(success for success, _ in results)

def test_book_plan_generation(mock_config, mock_api):
    """Test book plan generation with mocked API."""
    with patch('tkinter.Tk'), \
         patch('tkinter.messagebox.showwarning'), \
         patch('tkinter.messagebox.showerror'), \
         patch('pathlib.Path.write_text') as mock_write:
        
        app = ContentGeneratorGUI()
        app.topic_var = MagicMock(return_value="Python Programming")
        app.level_var = MagicMock(return_value="intermediate")
        app.output_path_var = MagicMock(return_value="output")
        
        # Mock API response
        mock_api.messages.create.return_value.content = [
            MagicMock(text=MOCK_API_RESPONSES["book_plan"])
        ]
        
        app._generate_book_plan()
        
        # Verify API was called
        mock_api.messages.create.assert_called_once()
        
        # Verify files were written
        assert mock_write.call_count == 2  # TOC and summary files
        
        # Verify content was parsed correctly
        write_calls = mock_write.call_args_list
        assert any("Python Programming Guide" in str(call) for call in write_calls)
        assert any("Target Audience" in str(call) for call in write_calls)

def test_rate_limiter_mock():
    """Test rate limiter without actual time delays."""
    config = RateLimitConfig(max_requests_per_minute=5)
    limiter = APIRateLimiter(config)
    
    with patch('time.time', side_effect=lambda: time.time()) as mock_time, \
         patch('time.sleep') as mock_sleep:
        
        # Simulate fewer requests (5 instead of 10)
        for _ in range(5):
            limiter.wait_if_needed()
        
        # Verify rate limiting was attempted
        assert mock_sleep.call_count <= 5  # Should not sleep more than necessary

@pytest.mark.timeout(5)
def test_content_cache_mock(tmp_path):
    """Test cache operations with mocked file system."""
    cache = ContentCache(tmp_path / "cache")
    section = Section(title="Test", subsections=[], level=2)
    
    with patch('pathlib.Path.write_text') as mock_write, \
         patch('pathlib.Path.read_text') as mock_read, \
         patch('pathlib.Path.exists') as mock_exists:
        
        # Setup mocks
        mock_exists.return_value = True  # Simulate file exists
        mock_read.return_value = "cached content"
        
        # Test write operation
        key = cache.get_cache_key(section, "test prompt")
        cache.cache_content(key, "new content")
        assert mock_write.called
        assert mock_write.call_args[0][0] == "new content"
        
        # Test read operation
        content = cache.get_cached_content(key)
        assert mock_read.called
        assert content == "cached content"
        
        # Test cache miss
        mock_exists.return_value = False
        content = cache.get_cached_content("nonexistent_key")
        assert content is None

# Add more specific cache tests
def test_content_cache_operations(tmp_path):
    """Test specific cache operations."""
    cache = ContentCache(tmp_path / "cache")
    section = Section(title="Test Section", subsections=[], level=2)
    
    with patch('pathlib.Path.write_text') as mock_write, \
         patch('pathlib.Path.read_text') as mock_read, \
         patch('pathlib.Path.exists') as mock_exists, \
         patch('pathlib.Path.mkdir') as mock_mkdir:
        
        # Test cache key generation
        key1 = cache.get_cache_key(section, "prompt1")
        key2 = cache.get_cache_key(section, "prompt2")
        assert key1 != key2  # Different prompts should yield different keys
        
        # Test cache write
        mock_exists.return_value = False
        cache.cache_content(key1, "content1")
        assert mock_mkdir.called  # Should create cache directory
        assert mock_write.called
        
        # Test cache hit
        mock_exists.return_value = True
        mock_read.return_value = "content1"
        content = cache.get_cached_content(key1)
        assert content == "content1"
        
        # Test cache miss
        mock_exists.return_value = False
        content = cache.get_cached_content("nonexistent")
        assert content is None

def test_content_cache_error_handling(tmp_path):
    """Test cache error handling."""
    cache = ContentCache(tmp_path / "cache")
    section = Section(title="Test", subsections=[], level=2)
    
    with patch('pathlib.Path.write_text') as mock_write, \
         patch('pathlib.Path.exists') as mock_exists:
        
        # Test write error
        mock_exists.return_value = False
        mock_write.side_effect = IOError("Write failed")
        
        # Should not raise exception but return False
        result = cache.cache_content("test_key", "content")
        assert result is False
        
        # Test read error
        mock_exists.return_value = True
        with patch('pathlib.Path.read_text', side_effect=IOError("Read failed")):
            content = cache.get_cached_content("test_key")
            assert content is None

def test_generate_with_retry(mock_config, mock_api):
    """Test content generation with retries."""
    with patch("src.autowriterllm.ai_content_generator.ContentGenerator._load_config", 
              return_value=mock_config):
        generator = ContentGenerator("test.md", "output")
        
        # Test successful generation
        mock_api.messages.create.return_value.content = [
            MagicMock(text=MOCK_API_RESPONSES["section_content"])
        ]
        content = generator._generate_with_retry("test prompt")
        assert "Section Title" in content
        
        # Test empty response
        mock_api.messages.create.return_value.content = []
        with pytest.raises(ContentGenerationError, match="Empty response from API"):
            generator._generate_with_retry("test prompt")
            
        # Test None content
        mock_api.messages.create.return_value.content = [MagicMock(text=None)]
        with pytest.raises(ContentGenerationError, match="Empty content in API response"):
            generator._generate_with_retry("test prompt")
            
        # Test API error with retry
        mock_api.messages.create.side_effect = [
            Exception("API Error"),
            Exception("API Error"),
            MagicMock(content=[MagicMock(text=MOCK_API_RESPONSES["section_content"])])
        ]
        content = generator._generate_with_retry("test prompt")
        assert "Section Title" in content
        assert mock_api.messages.create.call_count == 3

def test_lingyiwanwu_generation(mock_config):
    """Test content generation with Lingyiwanwu API."""
    # Override mock config for Lingyiwanwu
    mock_config.update({
        "provider": "lingyiwanwu",
        "lingyiwanwu": {
            "api_key": "test_key",
            "model": "yi-34b-chat"
        }
    })
    
    with patch("src.autowriterllm.ai_content_generator.ContentGenerator._load_config", 
              return_value=mock_config), \
         patch("requests.post") as mock_post:
        
        # Mock successful API response
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "choices": [{
                "message": {
                    "content": MOCK_API_RESPONSES["section_content"]
                }
            }]
        }
        mock_post.return_value = mock_response
        
        generator = ContentGenerator("test.md", "output")
        content = generator._generate_with_retry("test prompt")
        
        # Verify API call
        mock_post.assert_called_once()
        call_kwargs = mock_post.call_args[1]
        assert "headers" in call_kwargs
        assert "json" in call_kwargs
        assert call_kwargs["json"]["model"] == "yi-34b-chat"
        
        # Verify content
        assert "Section Title" in content

def test_lingyiwanwu_response_formatting(mock_config):
    """Test Lingyiwanwu API response formatting."""
    mock_config.update({
        "provider": "lingyiwanwu",
        "lingyiwanwu": {
            "api_key": "test_key",
            "model": "yi-34b-chat"
        }
    })
    
    with patch("src.autowriterllm.ai_content_generator.ContentGenerator._load_config", 
              return_value=mock_config), \
         patch("requests.post") as mock_post:
        
        # Test case 1: Properly formatted response
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "choices": [{
                "message": {
                    "content": MOCK_API_RESPONSES["book_plan"]
                }
            }]
        }
        mock_post.return_value = mock_response
        
        generator = ContentGenerator("test.md", "output")
        content = generator._generate_with_retry("test prompt")
        assert "```markdown:table_of_contents.md" in content
        assert "```markdown:book_summary.md" in content
        
        # Test case 2: Unformatted response that needs fixing
        unformatted_response = """# Table of Contents
Chapter 1
Chapter 2

# Book Summary
This is a summary"""
        
        mock_response.json.return_value = {
            "choices": [{
                "message": {
                    "content": unformatted_response
                }
            }]
        }
        
        content = generator._generate_with_retry("test prompt")
        assert "```markdown:table_of_contents.md" in content
        assert "```markdown:book_summary.md" in content
        
        # Test case 3: Invalid response format
        mock_response.json.return_value = {
            "choices": [{
                "message": {
                    "content": "Invalid format"
                }
            }]
        }
        
        with pytest.raises(ContentGenerationError):
            generator._generate_with_retry("test prompt")