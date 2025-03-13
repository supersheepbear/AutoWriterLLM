import pytest
from unittest.mock import MagicMock, patch
import ast
from src.autowriterllm.code_quality import ContentQualityAnalyzer, ContentAnalysis, CodeAnalysis

@pytest.fixture
def mock_ai_client():
    """Mock AI client for testing."""
    return MagicMock()

def test_analyze_content_with_different_types(mock_ai_client):
    """Test analyzing content with different content types."""
    analyzer = ContentQualityAnalyzer(mock_ai_client)
    
    # Mock the internal methods to return predictable values
    analyzer._analyze_accuracy = MagicMock(return_value=0.8)
    analyzer._analyze_clarity = MagicMock(return_value=0.7)
    analyzer._analyze_completeness = MagicMock(return_value=0.9)
    analyzer._analyze_engagement = MagicMock(return_value=0.6)
    analyzer._generate_improvements = MagicMock(return_value=["Suggestion 1", "Suggestion 2"])
    analyzer._identify_issues = MagicMock(return_value=["Issue 1", "Issue 2"])
    
    # Test with generic content type
    content = "This is test content for analysis."
    result = analyzer.analyze_content(content, "generic")
    
    # Verify the result
    assert isinstance(result, ContentAnalysis)
    assert result.accuracy_score == 0.8
    assert result.clarity_score == 0.7
    assert result.completeness_score == 0.9
    assert result.engagement_score == 0.6
    assert len(result.suggestions) == 2
    assert len(result.issues) == 2
    
    # Verify method calls with correct content type
    analyzer._analyze_accuracy.assert_called_with(content, "generic")
    analyzer._analyze_completeness.assert_called_with(content, "generic")
    analyzer._generate_improvements.assert_called_with(content, "generic")
    analyzer._identify_issues.assert_called_with(content, "generic")

def test_analyze_programming_content(mock_ai_client):
    """Test analyzing programming content with code blocks."""
    analyzer = ContentQualityAnalyzer(mock_ai_client)
    
    # Mock the internal methods
    analyzer._analyze_accuracy = MagicMock(return_value=0.8)
    analyzer._analyze_clarity = MagicMock(return_value=0.7)
    analyzer._analyze_completeness = MagicMock(return_value=0.9)
    analyzer._analyze_engagement = MagicMock(return_value=0.6)
    analyzer._generate_improvements = MagicMock(return_value=["Content suggestion"])
    analyzer._identify_issues = MagicMock(return_value=["Content issue"])
    
    # Mock code analysis
    code_analysis = CodeAnalysis(
        complexity_score=0.5,
        best_practices_score=0.6,
        documentation_score=0.7,
        type_hints_score=0.8,
        suggestions=["Code suggestion"],
        issues=["Code issue"]
    )
    analyzer._analyze_code = MagicMock(return_value=code_analysis)
    analyzer._extract_code_blocks = MagicMock(return_value=["def test(): pass"])
    
    # Test with programming content
    content = """
    # Programming Example
    
    Here's a code example:
    
    ```python
    def test():
        pass
    ```
    """
    result = analyzer.analyze_content(content, "programming")
    
    # Verify the result
    assert isinstance(result, ContentAnalysis)
    assert result.accuracy_score == 0.8
    assert result.clarity_score == 0.7
    assert result.completeness_score == 0.9
    assert result.engagement_score == 0.6
    
    # Verify that code suggestions and issues are included
    assert len(result.suggestions) == 2  # 1 content + 1 code
    assert "Code suggestion" in result.suggestions
    assert "Content suggestion" in result.suggestions
    
    assert len(result.issues) == 2  # 1 content + 1 code
    assert "Code issue" in result.issues
    assert "Content issue" in result.issues
    
    # Verify code extraction and analysis
    analyzer._extract_code_blocks.assert_called_once_with(content)
    analyzer._analyze_code.assert_called_once()

def test_error_handling_in_content_analysis(mock_ai_client):
    """Test error handling in content analysis."""
    analyzer = ContentQualityAnalyzer(mock_ai_client)
    
    # Make one of the internal methods raise an exception
    analyzer._analyze_accuracy = MagicMock(side_effect=Exception("Analysis error"))
    
    # Verify that the error is properly propagated
    with pytest.raises(Exception, match="Analysis error"):
        analyzer.analyze_content("content", "generic")

def test_code_analysis_error_handling(mock_ai_client):
    """Test error handling in code analysis."""
    analyzer = ContentQualityAnalyzer(mock_ai_client)
    
    # Make the code parsing raise an exception
    analyzer._extract_code_blocks = MagicMock(return_value=["invalid code"])
    
    # Mock other methods to avoid errors
    analyzer._analyze_accuracy = MagicMock(return_value=0.8)
    analyzer._analyze_clarity = MagicMock(return_value=0.7)
    analyzer._analyze_completeness = MagicMock(return_value=0.9)
    analyzer._analyze_engagement = MagicMock(return_value=0.6)
    analyzer._generate_improvements = MagicMock(return_value=[])
    analyzer._identify_issues = MagicMock(return_value=[])
    
    # Test with programming content
    result = analyzer.analyze_content("content with invalid code", "programming")
    
    # Verify that the analysis still completes despite code analysis errors
    assert isinstance(result, ContentAnalysis)
    assert result.accuracy_score == 0.8
    assert result.clarity_score == 0.7
    assert result.completeness_score == 0.9
    assert result.engagement_score == 0.6

def test_extract_code_blocks():
    """Test extracting code blocks from markdown content."""
    analyzer = ContentQualityAnalyzer(MagicMock())
    
    content = """
    # Test Content
    
    Here's a code example:
    
    ```python
    def example():
        return "Hello, world!"
    ```
    
    And another one:
    
    ```python
    class Test:
        def __init__(self):
            self.value = 42
    ```
    
    This is not a code block:
    
    ```
    Plain text
    ```
    """
    
    code_blocks = analyzer._extract_code_blocks(content)
    
    # Verify that only Python code blocks are extracted
    assert len(code_blocks) == 2
    assert "def example():" in code_blocks[0]
    assert "class Test:" in code_blocks[1]

def test_backward_compatibility():
    """Test backward compatibility with CodeQualityAnalyzer."""
    from src.autowriterllm.code_quality import CodeQualityAnalyzer
    
    analyzer = CodeQualityAnalyzer(MagicMock())
    
    # Mock the internal methods
    analyzer._analyze_code = MagicMock(return_value=CodeAnalysis(
        complexity_score=0.5,
        best_practices_score=0.6,
        documentation_score=0.7,
        type_hints_score=0.8,
        suggestions=["Suggestion"],
        issues=["Issue"]
    ))
    analyzer._extract_code_blocks = MagicMock(return_value=["def test(): pass"])
    
    # Test the analyze_code method
    result = analyzer.analyze_code("content with code")
    
    # Verify that the result is a CodeAnalysis object
    assert isinstance(result, CodeAnalysis)
    assert result.complexity_score == 0.5
    assert result.best_practices_score == 0.6
    assert result.documentation_score == 0.7
    assert result.type_hints_score == 0.8
    assert len(result.suggestions) == 1
    assert len(result.issues) == 1 