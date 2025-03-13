"""Custom exceptions for the autowriterllm package."""

class ContentGenerationError(Exception):
    """Exception raised for errors in the content generation process."""
    pass

class ReviewError(Exception):
    """Exception raised for errors in the content review process."""
    pass

class GenerationError(Exception):
    """Exception raised for errors in interactive content generation."""
    pass 