"""Data models for AutoWriterLLM.

This module contains the data models used throughout the AutoWriterLLM application.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
import time


@dataclass
class Section:
    """Represents a section in the table of contents.

    Attributes:
        title (str): The title of the section
        subsections (List[str]): List of subsection titles
        level (int): The heading level (1 for main sections, 2 for subsections)
        parent (Optional[str]): Parent section title if it's a subsection
    """
    title: str
    subsections: List[str]
    level: int
    parent: Optional[str] = None


@dataclass
class ChapterSummary:
    """Stores summary information for a chapter.

    Attributes:
        title (str): Chapter title
        summary (str): AI-generated summary of the chapter content
        timestamp (float): When the summary was generated
    """
    title: str
    summary: str
    timestamp: float = field(default_factory=time.time)


@dataclass
class GenerationProgress:
    """Track generation progress and statistics.
    
    Attributes:
        total_sections (int): Total number of sections to generate
        completed_sections (int): Number of completed sections
        failed_sections (int): Number of failed sections
        start_time (float): When generation started
    """
    total_sections: int
    completed_sections: int = 0
    failed_sections: int = 0
    start_time: float = field(default_factory=time.time)
    
    @property
    def progress_percentage(self) -> float:
        """Calculate progress percentage."""
        return (self.completed_sections / self.total_sections) * 100
    
    @property
    def estimated_time_remaining(self) -> float:
        """Estimate remaining time based on current progress."""
        if self.completed_sections == 0:
            return float('inf')
        
        elapsed_time = time.time() - self.start_time
        avg_time_per_section = elapsed_time / self.completed_sections
        remaining_sections = self.total_sections - self.completed_sections
        
        return avg_time_per_section * remaining_sections


@dataclass
class RateLimitConfig:
    """Configuration for API rate limiting.
    
    Attributes:
        max_requests_per_minute (int): Maximum API requests per minute
        retry_attempts (int): Number of retry attempts
        base_delay (float): Base delay between retries in seconds
    """
    max_requests_per_minute: int = 50
    retry_attempts: int = 3
    base_delay: float = 1.0  # seconds


@dataclass
class ConceptNode:
    """Represents a concept in the knowledge graph.
    
    Attributes:
        name (str): Name of the concept
        description (str): Description of the concept
        prerequisites (List[str]): List of prerequisite concepts
        related_concepts (List[str]): List of related concepts
    """
    name: str
    description: str
    prerequisites: List[str]
    related_concepts: List[str]


@dataclass
class ModelConfig:
    """Configuration for an LLM model.
    
    Attributes:
        name (str): Name of the model
        description (str): Description of the model
        max_tokens (int): Maximum tokens per request
        temperature (float): Temperature for generation
    """
    name: str
    description: str
    max_tokens: int = 8192
    temperature: float = 0.7


@dataclass
class ProviderConfig:
    """Configuration for an LLM provider.
    
    Attributes:
        api_key (str): API key for the provider
        base_url (str): Base URL for the provider's API
        models (List[ModelConfig]): List of available models
    """
    api_key: str
    base_url: str
    models: List[ModelConfig] = field(default_factory=list)


@dataclass
class AppConfig:
    """Application configuration.
    
    Attributes:
        provider (str): Default provider to use
        providers (Dict[str, ProviderConfig]): Available providers
        multi_part_generation (bool): Whether to generate content in multiple parts
        parts_per_section (int): Number of parts per section
        max_tokens_per_part (int): Maximum tokens per part
        include_learning_metadata (bool): Whether to include learning metadata
        include_previous_chapter_context (bool): Whether to include previous chapter context
        include_previous_section_context (bool): Whether to include previous section context
        output_dir (str): Output directory
        cache_dir (str): Cache directory
    """
    provider: str = "anthropic"
    providers: Dict[str, ProviderConfig] = field(default_factory=dict)
    multi_part_generation: bool = True
    parts_per_section: int = 5
    max_tokens_per_part: int = 8192
    include_learning_metadata: bool = True
    include_previous_chapter_context: bool = True
    include_previous_section_context: bool = True
    output_dir: str = "output"
    cache_dir: str = "cache" 