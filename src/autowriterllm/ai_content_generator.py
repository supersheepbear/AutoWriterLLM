import logging
from logging.handlers import RotatingFileHandler
import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Callable, Any
import tkinter as tk
from tkinter import ttk, scrolledtext, filedialog, messagebox
import anthropic
import markdown2
import yaml
from dataclasses import dataclass, field
import threading
import re
from tqdm import tqdm
import requests
import hashlib
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
from unittest.mock import MagicMock
import sys
from openai import OpenAI
import httpx

# Create logs directory if it doesn't exist
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)

# Create a unique log file for each session
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = log_dir / f"book_generation_{timestamp}.log"

# Configure logging with both file and console handlers
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        RotatingFileHandler(
            log_file,
            maxBytes=10485760,  # 10MB
            backupCount=5,
            encoding='utf-8'
        ),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)
logger.info(f"Starting new session, logging to: {log_file}")


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
class GenerationProgress:
    """Track generation progress and statistics."""
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
    """Configuration for API rate limiting."""
    max_requests_per_minute: int = 50
    retry_attempts: int = 3
    base_delay: float = 1.0  # seconds


class APIRateLimiter:
    """Handles API rate limiting and retries."""
    def __init__(self, config: RateLimitConfig):
        self.config = config
        self.request_times: List[float] = []
        self._lock = threading.Lock()

    def wait_if_needed(self) -> None:
        """Wait if we're exceeding rate limits."""
        with self._lock:
            now = time.time()
            # Remove old requests
            self.request_times = [t for t in self.request_times 
                                if now - t < 60]
            
            if len(self.request_times) >= self.config.max_requests_per_minute:
                sleep_time = 60 - (now - self.request_times[0])
                if sleep_time > 0:
                    time.sleep(sleep_time)
            
            self.request_times.append(now)


class ContentCache:
    """Cache for generated content to avoid redundant API calls."""
    
    def __init__(self, cache_dir: Path):
        """Initialize the cache.
        
        Args:
            cache_dir: Directory to store cache files
        """
        self.cache_dir = Path(cache_dir)
        try:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            logger.error(f"Failed to create cache directory: {e}")
            raise
        
    def get_cache_key(self, section: Section, prompt: str) -> str:
        """Generate unique cache key.
        
        Args:
            section: Section object
            prompt: Generation prompt
            
        Returns:
            str: MD5 hash of section title and prompt
        """
        content = f"{section.title}:{prompt}"
        return hashlib.md5(content.encode()).hexdigest()
        
    def get_cached_content(self, key: str) -> Optional[str]:
        """Retrieve cached content if available.
        
        Args:
            key: Cache key
            
        Returns:
            Optional[str]: Cached content or None if not found
        """
        try:
            cache_file = self.cache_dir / f"{key}.md"
            if cache_file.exists():
                return cache_file.read_text()
            return None
        except Exception as e:
            logger.error(f"Error reading from cache: {e}")
            return None
        
    def cache_content(self, key: str, content: str) -> bool:
        """Cache generated content.
        
        Args:
            key: Cache key
            content: Content to cache
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            cache_file = self.cache_dir / f"{key}.md"
            cache_file.write_text(content)
            logger.debug(f"Content cached successfully: {key}")
            return True
        except Exception as e:
            logger.error(f"Failed to cache content: {e}")
            return False


class ContentGenerator:
    """Handles the generation of content using AI APIs.

    This class manages the parsing of the table of contents and generation
    of content using AI APIs (Claude/Anthropic API in this implementation).

    Attributes:
        toc_file (Path): Path to the table of contents markdown file
        output_dir (Path): Directory where generated content will be saved
        api_key (str): API key for the AI service
        sections (List[Section]): Parsed sections from the table of contents

    Example:
        >>> generator = ContentGenerator("table_of_contents.md", "output")
        >>> generator.parse_toc()
        >>> generator.generate_all_content()
    """

    def __init__(self, toc_file: Union[str, Path], output_dir: Union[str, Path]):
        """Initialize the ContentGenerator.

        Args:
            toc_file: Path to the table of contents markdown file
            output_dir: Directory where generated content will be saved

        Raises:
            FileNotFoundError: If the table of contents file doesn't exist
            PermissionError: If the output directory can't be created/accessed
        """
        self.toc_file = Path(toc_file)
        self.output_dir = Path(output_dir)
        self.sections: List[Section] = []

        # Load config and initialize API client
        self.config = self._load_config()
        self.client = self._initialize_client()

        # Initialize cache and rate limiter
        self.cache = ContentCache(self.output_dir / "cache")
        self.rate_limiter = APIRateLimiter(RateLimitConfig())
        self.validator = ContentValidator()

        # Create output directory if it doesn't exist
        try:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created/verified output directory: {self.output_dir}")
        except PermissionError as e:
            logger.error(f"Failed to create output directory: {e}")
            raise

        logger.debug(f"ContentGenerator initialized with TOC: {self.toc_file}, Output: {self.output_dir}")

        # Add progress window as class attribute
        self.progress_window: Optional[tk.Toplevel] = None
        self.progress_var: Optional[tk.DoubleVar] = None
        self.attempt_label: Optional[ttk.Label] = None
        self.time_label: Optional[ttk.Label] = None

    def _load_config(self) -> Dict:
        """Load configuration from config file.

        Returns:
            Dict: Configuration dictionary

        Raises:
            ValueError: If configuration is invalid
        """
        try:
            config_path = Path("config.yaml")
            if config_path.exists():
                with open(config_path) as f:
                    config = yaml.safe_load(f)
                    if config.get("provider") not in ["anthropic", "lingyiwanwu"]:
                        raise ValueError("Invalid provider specified in config")
                    return config
            raise ValueError("Config file not found")
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            raise

    def _initialize_client(self) -> Union[anthropic.Anthropic, Any]:
        """Initialize the appropriate API client based on configuration."""
        provider = self.config.get("provider")
        try:
            if provider == "anthropic":
                api_key = self.config.get("anthropic", {}).get("api_key")
                if not api_key:
                    raise ValueError("Anthropic API key not found in config")
                return anthropic.Anthropic(api_key=api_key)
            elif provider == "lingyiwanwu":
                api_key = self.config.get("lingyiwanwu", {}).get("api_key")
                if not api_key:
                    raise ValueError("Lingyiwanwu API key not found in config")
                
                # Configure custom httpx client with longer timeouts
                http_client = httpx.Client(
                    timeout=httpx.Timeout(
                        connect=10.0,    # connection timeout
                        read=300.0,      # read timeout
                        write=20.0,      # write timeout
                        pool=10.0        # pool timeout
                    ),
                    limits=httpx.Limits(
                        max_keepalive_connections=5,
                        max_connections=10,
                        keepalive_expiry=30.0
                    )
                )
                
                # Initialize OpenAI client with custom configuration
                return OpenAI(
                    api_key=api_key,
                    base_url="https://api.lingyiwanwu.com/v1",
                    http_client=http_client,
                    timeout=300.0
                )
            else:
                raise ValueError(f"Unsupported provider: {provider}")
        except Exception as e:
            logger.error(f"Failed to initialize API client: {e}")
            raise

    def parse_toc(self) -> None:
        """Parse the table of contents markdown file.
        
        Parses a markdown file containing a table of contents and extracts sections and subsections.
        The expected format is:
        ## Main Section
        ### Subsection
        
        Raises:
            FileNotFoundError: If TOC file doesn't exist
            ValueError: If TOC file is empty or malformed
        """
        logger.info("Starting to parse table of contents...")
        try:
            if not self.toc_file.exists():
                logger.error(f"TOC file does not exist: {self.toc_file}")
                raise FileNotFoundError(f"TOC file not found: {self.toc_file}")

            content = self.toc_file.read_text()
            if not content.strip():
                logger.warning("TOC file is empty or only contains whitespace.")
                raise ValueError("TOC file is empty")

            current_section = None
            self.sections = []

            for line in content.split("\n"):
                line = line.strip()
                if not line:
                    continue

                # Count leading '#' to determine level
                level = 0
                for char in line:
                    if char == '#':
                        level += 1
                    else:
                        break

                if level == 0:
                    continue

                title = line.lstrip('#').strip()

                if level == 2:  # Main section
                    current_section = Section(title=title, subsections=[], level=level)
                    self.sections.append(current_section)
                    logger.debug(f"Added main section: {title}")

                elif level == 3 and current_section:  # Subsection
                    current_section.subsections.append(title)
                    logger.debug(f"Added subsection: {title} to {current_section.title}")

            if not self.sections:
                logger.warning("No sections found in TOC file")
                raise ValueError("No valid sections found in TOC file")

            logger.info(f"Successfully parsed {len(self.sections)} sections")

        except Exception as e:
            logger.error(f"Error parsing TOC file: {str(e)}")
            raise

    def generate_section_content(self, section: Section) -> Tuple[str, str]:
        """Generate content for a section and all its subsections sequentially.

        Args:
            section: Section object containing title and subsections

        Returns:
            Tuple[str, str]: Generated content and filename

        Raises:
            ContentGenerationError: If content generation fails after all retries
        """
        logger.info(f"Generating content for main section: {section.title}")
        
        # Generate main section content
        main_content = []
        try:
            # Generate main section introduction
            intro_prompt = self._create_prompt(section, is_introduction=True)
            intro_content = self._generate_with_retry(intro_prompt)
            main_content.append(f"# {section.title}\n\n{intro_content}\n")
            logger.info(f"Generated introduction for {section.title}")
            
            # Generate each subsection sequentially
            for subsection in section.subsections:
                logger.info(f"Generating content for subsection: {subsection}")
                subsection_obj = Section(
                    title=subsection,
                    subsections=[],
                    level=3,
                    parent=section.title
                )
                
                try:
                    # Generate subsection content
                    subsection_prompt = self._create_prompt(subsection_obj)
                    subsection_content = self._generate_with_retry(subsection_prompt)
                    main_content.append(f"## {subsection}\n\n{subsection_content}\n")
                    logger.info(f"Successfully generated content for subsection: {subsection}")
                    
                except Exception as e:
                    logger.error(f"Failed to generate subsection {subsection}: {e}")
                    raise ContentGenerationError(f"Failed to generate subsection {subsection}: {e}")
                
            # Combine all content
            full_content = "\n".join(main_content)
            return full_content, self._create_filename(section)
            
        except Exception as e:
            logger.error(f"Failed to generate content for section {section.title}: {e}")
            raise ContentGenerationError(f"Failed to generate section {section.title}: {e}")

    def _get_context_using_vectors(self, section: Section) -> str:
        """Get context using vector similarity search."""
        try:
            import chromadb
            from chromadb.config import Settings

            # Create persistent vector DB path
            db_path = self.output_dir / "vector_dbs"
            db_path.mkdir(parents=True, exist_ok=True)
            logger.info(f"Using vector database at: {db_path}")

            # Initialize ChromaDB with persistent storage
            client = chromadb.Client(
                Settings(
                    is_persistent=True,
                    persist_directory=str(db_path),
                    anonymized_telemetry=False,
                )
            )

            # Use consistent collection name
            collection_name = "sections"
            
            # Create or get collection
            try:
                collection = client.get_collection(collection_name)
                logger.info(f"Using existing collection: {collection_name}")
            except Exception:
                logger.info(f"Creating new collection: {collection_name}")
                collection = client.create_collection(collection_name)

            # Get current section index and level
            current_index = next(
                (i for i, s in enumerate(self.sections) if s.title == section.title), -1
            )
            
            # For subsections, include the parent section's content
            if section.parent:
                parent_index = next(
                    (i for i, s in enumerate(self.sections) if s.title == section.parent), -1
                )
                if parent_index >= 0:
                    logger.info(f"Found parent section: {section.parent}")
                    parent_file = self.output_dir / self._create_filename(self.sections[parent_index])
                    if parent_file.exists():
                        parent_content = parent_file.read_text()
                        collection.add(
                            documents=[parent_content],
                            metadatas=[{"title": section.parent}],
                            ids=[f"parent_{section.parent}"]
                        )
                        logger.info(f"Added parent section content to context: {section.parent}")

            # Only index sections that come before the current one
            if current_index > 0:
                # Get list of existing document IDs
                try:
                    existing_ids = {doc["id"] for doc in collection.get()["ids"]}
                except Exception:
                    existing_ids = set()
                    logger.debug("No existing documents found")

                # Index previous sections if not already indexed
                indexed_count = 0
                for prev_section in self.sections[:current_index]:
                    section_id = f"{prev_section.title}"
                    
                    if section_id not in existing_ids:
                        # Generate content for indexing if file exists
                        file_path = self.output_dir / self._create_filename(prev_section)
                        if file_path.exists():
                            content = file_path.read_text()
                            collection.add(
                                documents=[content],
                                metadatas=[{"title": prev_section.title}],
                                ids=[section_id]
                            )
                            indexed_count += 1
                            logger.info(f"Indexed previous section: {prev_section.title}")

                logger.info(f"Indexed {indexed_count} previous sections for context search")

                # Only query if we have indexed documents
                if indexed_count > 0:
                    results = collection.query(
                        query_texts=[section.title],
                        n_results=min(2, indexed_count)
                    )

                    if results and results["documents"] and results["documents"][0]:
                        context_sections = []
                        for doc, metadata in zip(results["documents"][0], results["metadatas"][0]):
                            # Extract key points (last 500 chars) from each relevant section
                            context = f"\nContext from {metadata['title']}:\n{doc[-500:]}"
                            context_sections.append(context)
                            logger.debug(f"Added context from: {metadata['title']}")
                        
                        return "\n\n".join(context_sections)
                    
                    logger.info("No relevant context found in previous sections")
                    return ""
                else:
                    logger.info("No previous sections available for context")
                    return ""
            else:
                logger.info("First section - no previous context needed")
                return ""

        except Exception as e:
            logger.error(f"Error getting vector context: {e}", exc_info=True)
            return ""

    def _determine_tutorial_type(self) -> str:
        """Determine the tutorial context from the summary file.

        Looks for context in the summary file that should be in the same directory
        as the table of contents file, with the same name but _summary suffix.

        Returns:
            str: Description of the tutorial context
        """
        try:
            # Construct summary file path
            summary_file = self.toc_file.parent / f"{self.toc_file.stem}_summary.md"
            
            if not summary_file.exists():
                logger.warning(f"Summary file not found: {summary_file}")
                return ""

            content = summary_file.read_text()
            logger.info(f"Found summary file: {summary_file}")
            
            # Return the entire summary content as context
            return content.strip()

        except Exception as e:
            logger.warning(f"Error reading summary file: {e}")
            return ""

    def _create_prompt(self, section: Section, is_introduction: bool = False) -> str:
        """Create a prompt with proper context."""
        # Get context from previous sections
        vector_context = self._get_context_using_vectors(section)
        
        # Get tutorial type and summary context
        tutorial_context = self._determine_tutorial_type()
        
        # Combine contexts
        context = f"""Tutorial Type: {tutorial_context}

Previous Content Context:
{vector_context}"""

        if is_introduction:
            prompt = f"""Write an introduction for the section '{section.title}'.

{context}

This introduction should:
1. Provide an overview of {section.title}
2. Explain why this topic is important
3. Build upon concepts from previous sections where relevant
4. Outline what will be covered in the subsections:
{chr(10).join(f'- {sub}' for sub in section.subsections)}

Format the response in markdown."""
        else:
            prompt = f"""Write a detailed tutorial subsection about '{section.title}'.
Parent section: {section.parent}

{context}

Requirements for code examples:
1. Use Python 3.12 and follow Python community best practices
2. Include comprehensive logging and error handling
3. Handle edge cases, invalid inputs, and unusual scenarios
4. Optimize performance while maintain clarity
5. Use type hints for all variables and functions
6. Provide Google-style docstrings with examples for Sphinx documentation

For each code example:
- Include detailed explanations of the code
- Explain design decisions and trade-offs
- Highlight best practices and common pitfalls
- Show both basic and advanced usage patterns
- Reference and build upon concepts from previous sections

At the end of the subsection, include:
1. Practice exercises specific to this subsection
2. Key takeaways and summary

Format the response in markdown, using appropriate headers and code blocks."""

        return prompt

    def _create_filename(self, section: Section) -> str:
        """Create a filename for the section content.
        
        Args:
            section: Section object
            
        Returns:
            str: Sanitized filename with proper extension
            
        Examples:
            Chapter 1: Introduction -> chapter-1.md
            1.1 What is PyQt? -> chapter-1-1.md
            1.1.1 Subsection -> chapter-1-1-1.md
        """
        try:
            # Extract chapter number
            if section.level == 2:  # Main chapter
                chapter_num = section.title.split(':')[0].strip().lower()
                chapter_num = re.sub(r'chapter\s*', '', chapter_num)
                return f"chapter-{chapter_num}.md"
            else:  # Subsection
                # Get chapter number from parent
                parent_num = section.parent.split(':')[0].strip().lower()
                parent_num = re.sub(r'chapter\s*', '', parent_num)
                
                # Get section number
                section_num = section.title.split(' ')[0].strip()
                
                # Count the number of dots to determine subsection depth
                depth = section_num.count('.') + 1
                
                # Create filename based on section numbering
                section_parts = section_num.split('.')
                filename = f"chapter-{'-'.join(section_parts)}.md"
                return filename
                
        except Exception as e:
            logger.error(f"Error creating filename for section {section.title}: {e}")
            # Fallback to sanitized title if parsing fails
            safe_name = re.sub(r'[^\w\s-]', '', section.title.lower())
            safe_name = re.sub(r'[-\s]+', '-', safe_name)
            return f"{safe_name}.md"

    def generate_all_content(
        self, 
        progress_callback: Optional[Callable[[float], None]] = None
    ) -> None:
        """Generate content for all sections sequentially, continuing from where it left off."""
        total_units = sum(len(section.subsections) + 1 for section in self.sections)
        completed_units = 0
        
        logger.info(f"Starting content generation for {len(self.sections)} sections ({total_units} total units)")
        
        for section in self.sections:
            chapter_file = self.output_dir / self._create_filename(section)
            if chapter_file.exists():
                logger.info(f"Skipping already generated section: {section.title}")
                completed_units += len(section.subsections) + 1
                if progress_callback:
                    progress = (completed_units / total_units) * 100
                    progress_callback(progress)
                continue
            
            try:
                logger.info(f"Processing section: {section.title}")
                
                # Generate and save main section introduction
                logger.info(f"Generating introduction for: {section.title}")
                intro_prompt = self._create_prompt(section, is_introduction=True)
                intro_content = self._generate_with_retry(intro_prompt)
                
                # Save main section to its own file
                self._safe_file_write(chapter_file, f"# {section.title}\n\n{intro_content}\n")
                logger.info(f"Saved chapter introduction to: {chapter_file}")
                
                # Add to vector DB for context
                self._add_to_vector_db(section.title, intro_content)
                
                completed_units += 1
                if progress_callback:
                    progress = (completed_units / total_units) * 100
                    progress_callback(progress)
                logger.info(f"Progress: {progress:.1f}% ({completed_units}/{total_units} units)")
                
                # Generate each subsection with updated context
                for subsection in section.subsections:
                    subsection_file = self.output_dir / self._create_filename(Section(title=subsection, subsections=[], level=3, parent=section.title))
                    if subsection_file.exists():
                        logger.info(f"Skipping already generated subsection: {subsection}")
                        completed_units += 1
                        if progress_callback:
                            progress = (completed_units / total_units) * 100
                            progress_callback(progress)
                        continue
                    
                    try:
                        logger.info(f"Generating content for subsection: {subsection}")
                        subsection_obj = Section(
                            title=subsection,
                            subsections=[],
                            level=3,
                            parent=section.title
                        )
                        
                        subsection_prompt = self._create_prompt(subsection_obj)
                        subsection_content = self._generate_with_retry(subsection_prompt)
                        
                        # Save subsection to its own file
                        self._safe_file_write(
                            subsection_file, 
                            f"# {subsection}\n\n{subsection_content}\n"
                        )
                        logger.info(f"Saved subsection to: {subsection_file}")
                        
                        # Update vector DB
                        self._add_to_vector_db(subsection, subsection_content)
                        
                        completed_units += 1
                        if progress_callback:
                            progress = (completed_units / total_units) * 100
                            progress_callback(progress)
                        logger.info(f"Progress: {progress:.1f}% ({completed_units}/{total_units} units)")
                        
                    except Exception as e:
                        logger.error(f"Failed to generate subsection {subsection}: {e}")
                        raise ContentGenerationError(f"Failed to generate subsection {subsection}: {e}")
                
            except Exception as e:
                logger.error(f"Failed to generate section {section.title}: {e}")
                raise ContentGenerationError(f"Failed to generate section {section.title}: {e}")
        
        logger.info(f"Content generation completed successfully. Generated {completed_units} units of content.")

    def _add_to_vector_db(self, doc_id: str, content: str) -> None:
        """Add or update content in vector database."""
        try:
            import chromadb
            from chromadb.config import Settings
            
            db_path = self.output_dir / "vector_dbs"
            db_path.mkdir(parents=True, exist_ok=True)
            
            client = chromadb.Client(
                Settings(
                    is_persistent=True,
                    persist_directory=str(db_path),
                    anonymized_telemetry=False
                )
            )
            
            collection_name = "sections"
            try:
                collection = client.get_collection(collection_name)
            except Exception:
                collection = client.create_collection(collection_name)
                
            # Delete existing document if it exists
            try:
                collection.delete(ids=[doc_id])
                logger.debug(f"Deleted existing document: {doc_id}")
            except Exception:
                pass
                
            # Add new content
            collection.add(
                documents=[content],
                metadatas=[{"title": doc_id}],
                ids=[doc_id]
            )
            logger.debug(f"Added content to vector database for section: {doc_id}")
            
        except Exception as e:
            logger.error(f"Failed to add content to vector database: {e}")

    def _create_book_planning_prompt(self, user_input: str) -> str:
        """Create a detailed prompt from minimal user input.
        
        Args:
            user_input: Basic user request for content generation
            
        Returns:
            str: Detailed prompt for the AI model
        """
        # Parse level and topic from user input
        level = "beginner"
        topic = user_input
        if "level" in user_input.lower():
            parts = user_input.lower().split("level")
            topic = parts[0].strip()
            level = parts[1].strip()

        system_prompt = """Please generate two markdown files in EXACTLY this format:

```markdown:table_of_contents.md
# [Topic Name]

## 1. [Chapter Title]
### 1.1 [Section Title]
### 1.2 [Section Title]

## 2. [Chapter Title]
### 2.1 [Section Title]
### 2.2 [Section Title]
[etc...]
```

```markdown:book_summary.md
# [Topic Name]

## Overview
[Overview content]

## Target Audience
- [Audience point 1]
- [Audience point 2]

## Prerequisites
- [Prerequisite 1]
- [Prerequisite 2]

## Learning Objectives
By the end of this guide, readers will be able to:
1. [Objective 1]
2. [Objective 2]

## Technical Requirements
- [Requirement 1]
- [Requirement 2]

## Estimated Completion Time
- Total Hours: [X]
- Recommended Pace: [Y]
```

The content should follow these requirements:
1. Progress from fundamentals to advanced concepts
2. Include practical examples and exercises
3. Cover best practices and common pitfalls
4. Include real-world applications
5. Provide clear learning objectives for each section"""

        prompt = f"""System: {system_prompt}

User Request: Create a {level} level guide about {topic}.

Please generate a comprehensive learning plan following the EXACT format shown above.
The content should be suitable for {level} learners while still providing clear progression
to advanced concepts."""

        return prompt

    def _parse_ai_response(self, content: str) -> Tuple[str, str]:
        """Parse AI response to separate TOC and summary content.
        
        Args:
            content: Raw AI response string
            
        Returns:
            Tuple[str, str]: (toc_content, summary_content)
            
        Raises:
            ValueError: If content cannot be properly parsed
        """
        try:
            # Look for markdown code blocks with or without file labels
            patterns = [
                # Try with file labels first
                (r"```markdown:table_of_contents\.md\n(.*?)```", r"```markdown:book_summary\.md\n(.*?)```"),
                # Fallback to simple markdown blocks
                (r"```markdown\n# table_of_contents\.md\n(.*?)```", r"```markdown\n# book_summary\.md\n(.*?)```"),
                # Another common format
                (r"```\n# table_of_contents\.md\n(.*?)```", r"```\n# book_summary\.md\n(.*?)```")
            ]
            
            for toc_pattern, summary_pattern in patterns:
                toc_match = re.search(toc_pattern, content, re.DOTALL)
                summary_match = re.search(summary_pattern, content, re.DOTALL)
                
                if toc_match and summary_match:
                    toc_content = toc_match.group(1).strip()
                    summary_content = summary_match.group(1).strip()
                    logger.debug("Successfully parsed AI response using pattern")
                    return toc_content, summary_content
            
            # If no patterns matched, try to find any markdown blocks
            markdown_blocks = re.findall(r"```(?:markdown)?\n(.*?)```", content, re.DOTALL)
            if len(markdown_blocks) >= 2:
                logger.warning("Using fallback pattern matching for markdown blocks")
                return markdown_blocks[0].strip(), markdown_blocks[1].strip()
            
            raise ValueError("Could not find properly formatted TOC or summary in AI response")
            
        except Exception as e:
            logger.error(f"Error parsing AI response: {e}")
            logger.debug(f"Raw content: {content[:500]}...")  # Log first 500 chars of content
            raise ValueError(f"Failed to parse AI response: {e}")

    def _create_retry_window(self) -> None:
        """Create progress window for retry attempts."""
        if self.progress_window is None:
            self.progress_window = tk.Toplevel()
            self.progress_window.title("API Retry Progress")
            self.progress_window.geometry("300x150")
            
            # Center the window
            if hasattr(self, 'root'):
                x = self.root.winfo_x() + self.root.winfo_width()//2 - 150
                y = self.root.winfo_y() + self.root.winfo_height()//2 - 75
                self.progress_window.geometry(f"+{x}+{y}")
            
            self.progress_var = tk.DoubleVar()
            
            ttk.Label(self.progress_window, text="Retrying API call...").pack(pady=10)
            progress_bar = ttk.Progressbar(
                self.progress_window, 
                variable=self.progress_var,
                maximum=10,
                length=200
            )
            progress_bar.pack(pady=10)
            
            self.attempt_label = ttk.Label(self.progress_window, text="Attempt 1/10")
            self.attempt_label.pack(pady=5)
            
            self.time_label = ttk.Label(self.progress_window, text="Waiting...")
            self.time_label.pack(pady=5)

    def _generate_with_retry(self, prompt: str, max_retries: int = 10) -> str:
        """Generate content with automatic retry on failure."""
        for attempt in range(max_retries):
            try:
                logger.info(f"Generation attempt {attempt + 1}/{max_retries}")
                
                if self.config["provider"] == "anthropic":
                    logger.debug("Using Anthropic API")
                    response = self.client.messages.create(
                        model=self.config["anthropic"]["model"],
                        max_tokens=4096,
                        temperature=0.7,
                        messages=[{"role": "user", "content": prompt}]
                    )
                    content = response.content[0].text
                    logger.info("Successfully generated content from Anthropic API")
                    return content
                    
                elif self.config["provider"] == "lingyiwanwu":
                    logger.debug("Using Lingyiwanwu API")
                    url = "https://api.lingyiwanwu.com/v1/chat/completions"
                    headers = {
                        "Authorization": f"Bearer {self.config['lingyiwanwu']['api_key']}",
                        "Content-Type": "application/json"
                    }
                    
                    data = {
                        "model": self.config["lingyiwanwu"]["model"],
                        "messages": [
                            {"role": "user", "content": prompt}
                        ],
                        "temperature": 0.7,
                        "max_tokens": 4096
                    }
                    
                    logger.debug(f"Request data: {data}")
                    response = requests.post(url, headers=headers, json=data, timeout=60)
                    
                    if response.status_code != 200:
                        logger.error(f"API error: {response.status_code} - {response.text}")
                        raise ContentGenerationError(f"API error: {response.status_code}")
                    
                    result = response.json()
                    logger.debug(f"API response: {result}")
                    
                    if not result.get("choices"):
                        logger.error("No choices in API response")
                        raise ContentGenerationError("No content in API response")
                    
                    content = result["choices"][0]["message"]["content"]
                    if not content:
                        logger.error("Empty content in API response")
                        raise ContentGenerationError("Empty content in API response")
                    
                    logger.info("Successfully generated content from Lingyiwanwu API")
                    return content
                    
            except (requests.Timeout, requests.ConnectionError) as e:
                logger.warning(f"Network error on attempt {attempt + 1}: {e}")
                if attempt == max_retries - 1:
                    raise ContentGenerationError(f"Network error after {max_retries} attempts")
                wait_time = min(300, 10 * (2 ** attempt))
                logger.info(f"Waiting {wait_time} seconds before retry...")
                time.sleep(wait_time)
                continue
                
            except Exception as e:
                logger.error(f"Error on attempt {attempt + 1}: {e}")
                if attempt == max_retries - 1:
                    raise ContentGenerationError(f"Failed after {max_retries} attempts: {str(e)}")
                wait_time = min(300, 10 * (2 ** attempt))
                logger.info(f"Waiting {wait_time} seconds before retry...")
                time.sleep(wait_time)
        
        raise ContentGenerationError("Failed to generate content after all retries")

    def __del__(self):
        """Cleanup on object destruction."""
        if hasattr(self, 'progress_window') and self.progress_window:
            self.progress_window.destroy()

    def _validate_config(self, config: Dict) -> None:
        """Validate configuration dictionary.
        
        Args:
            config: Configuration dictionary to validate
            
        Raises:
            ValueError: If configuration is invalid
        """
        required_fields = {
            "provider": str,
            "anthropic": {"api_key": str, "model": str},
            "lingyiwanwu": {"api_key": str, "model": str}
        }
        
        def validate_structure(data: Dict, template: Dict, path: str = "") -> None:
            for key, value_type in template.items():
                if key not in data:
                    raise ValueError(f"Missing required field: {path}{key}")
                
                if isinstance(value_type, dict):
                    if not isinstance(data[key], dict):
                        raise ValueError(f"Invalid type for {path}{key}")
                    validate_structure(data[key], value_type, f"{path}{key}.")
                elif not isinstance(data[key], value_type):
                    raise ValueError(f"Invalid type for {path}{key}")
        
        validate_structure(config, required_fields)

    def _validate_topic_input(self, topic: str) -> bool:
        """Validate topic input string.
        
        Args:
            topic (str): Topic string to validate
            
        Returns:
            bool: True if valid, False otherwise
            
        Note:
            Validates the topic string for:
            - Minimum length (3 characters)
            - Maximum length (500 characters)
            - Contains actual text characters
            - Not just special characters or whitespace
        """
        # Check if empty or too short
        if not topic or len(topic.strip()) < 3:
            messagebox.showwarning(
                "Invalid Input",
                "Topic description must be at least 3 characters long"
            )
            logger.warning("Topic validation failed: Too short")
            return False
        
        # Check for meaningful content (not just special characters)
        if not re.search(r'[a-zA-Z]', topic):
            messagebox.showwarning(
                "Invalid Input",
                "Topic must contain some text characters"
            )
            logger.warning("Topic validation failed: No text characters")
            return False
        
        # Check maximum length
        if len(topic) > 500:
            messagebox.showwarning(
                "Invalid Input",
                "Topic description is too long (max 500 characters)"
            )
            logger.warning("Topic validation failed: Too long")
            return False
        
        logger.debug(f"Topic validation passed: {topic}")
        return True

    def _safe_file_write(self, file_path: Path, content: str) -> bool:
        """Safely write content to file with error handling."""
        try:
            # Create backup of existing file if it exists
            if file_path.exists():
                backup_path = file_path.with_suffix(f".bak_{int(time.time())}")
                file_path.rename(backup_path)
                logger.info(f"Created backup: {backup_path}")
            
            # Write new content
            file_path.write_text(content)
            logger.info(f"Successfully wrote file: {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to write file {file_path}: {e}")
            return False


class ContentValidator:
    """Validates generated content quality."""
    
    @staticmethod
    def validate_content(content: str, section: Section) -> Tuple[bool, List[str]]:
        """Validate content meets quality standards."""
        issues = []
        
        # Check minimum length
        if len(content) < 1000:
            issues.append("Content length below minimum threshold")
            
        # Check for code blocks if technical section
        if "code" in section.title.lower():
            if "```" not in content:
                issues.append("Missing code examples")
                
        # Check for section headers
        if not re.search(r'^#{1,6}\s', content, re.MULTILINE):
            issues.append("Missing section headers")
            
        # Check for practice questions
        if "Practice Questions" not in content:
            issues.append("Missing practice questions section")
            
        return len(issues) == 0, issues


class ContentGeneratorGUI:
    """GUI interface for the ContentGenerator.

    Provides a simple interface to start content generation and view progress.

    Attributes:
        root (tk.Tk): Main window
        generator (ContentGenerator): Content generator instance
    """

    def __init__(self):
        """Initialize the GUI application."""
        self.root = tk.Tk()
        self.root.title("Content Generator")
        self.root.geometry("800x600")

        # Initialize path variables
        self.toc_path: Optional[Path] = None
        self.summary_path: Optional[Path] = None
        self.output_path: Optional[Path] = None

        # Initialize StringVars
        self.topic_var = tk.StringVar()
        self.level_var = tk.StringVar(value="Beginner")
        self.output_path_var = tk.StringVar()
        self.toc_path_var = tk.StringVar()
        self.summary_path_var = tk.StringVar()
        self.progress_var = tk.DoubleVar()

        # Available models configuration based on Lingyiwanwu API docs
        self.models = {
            "Anthropic": ["claude-3-sonnet-20240229", "claude-3-opus-20240229"],
            "Lingyiwanwu": [
                "yi-lightning",      # Latest high-performance model
                "yi-large",          # Large model with enhanced capabilities
                "yi-medium",         # Medium-sized balanced model
                "yi-vision",         # Complex vision task model
                "yi-medium-200k",    # 200K context window model
                "yi-spark",          # Small but efficient model
                "yi-large-rag",      # Real-time web retrieval model
                "yi-large-fc",       # Function calling support
                "yi-large-turbo",    # High performance-cost ratio
            ]
        }

        logger.debug("Initialized model options:")
        for provider, model_list in self.models.items():
            logger.debug(f"{provider} models: {model_list}")
        
        self._create_widgets()
        self._load_config()
        
        # Set up error handling
        self.root.report_callback_exception = self._handle_callback_error

        # Create default output directory
        self.default_output_dir = Path("output")
        self.default_output_dir.mkdir(exist_ok=True)
        self.output_path = self.default_output_dir
        self.output_path_var.set(str(self.default_output_dir))

        logger.debug("GUI initialized with default paths")

    def _handle_callback_error(self, exc_type, exc_value, exc_traceback):
        """Handle uncaught exceptions in callbacks."""
        error_msg = f"An error occurred: {exc_type.__name__}: {exc_value}"
        logger.error(error_msg, exc_info=(exc_type, exc_value, exc_traceback))
        messagebox.showerror("Error", error_msg)
        self._enable_inputs()  # Ensure UI is enabled

    def _create_widgets(self) -> None:
        """Create and arrange GUI widgets."""
        # Main container frame with padding
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Create a canvas and scrollable frame for content
        canvas = tk.Canvas(main_frame)
        scrollbar = ttk.Scrollbar(main_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        # Pack the canvas and scrollbar
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # Quick Topic Planning Section
        quick_plan_frame = ttk.LabelFrame(
            scrollable_frame, text="Quick Topic Planning", padding="10"
        )
        quick_plan_frame.pack(fill=tk.X, pady=(0, 10))

        # Topic Input
        topic_frame = ttk.Frame(quick_plan_frame)
        topic_frame.pack(fill=tk.X, pady=5)

        self.topic_label = ttk.Label(topic_frame, text="What do you want to learn:", width=20)
        self.topic_label.pack(side=tk.LEFT)

        self.topic_entry = ttk.Entry(topic_frame, textvariable=self.topic_var, width=50)
        self.topic_entry.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)

        # Level Selection
        level_frame = ttk.Frame(quick_plan_frame)
        level_frame.pack(fill=tk.X, pady=5)

        self.level_label = ttk.Label(level_frame, text="Your current level:", width=20)
        self.level_label.pack(side=tk.LEFT)

        self.level_combo = ttk.Combobox(
            level_frame,
            textvariable=self.level_var,
            values=["Beginner", "Intermediate", "Advanced"],
            state="readonly",
            width=20
        )
        self.level_combo.pack(side=tk.LEFT, padx=5)
        self.level_combo.set("Beginner")  # Default value

        # Generate Book Plan Button
        self.generate_plan_button = ttk.Button(
            quick_plan_frame,
            text="Generate Book Plan",
            command=self._generate_book_plan
        )
        self.generate_plan_button.pack(pady=5)

        # Top section - Model Configuration
        model_frame = ttk.LabelFrame(
            scrollable_frame, text="AI Model Configuration", padding="10"
        )
        model_frame.pack(fill=tk.X, pady=(0, 10))

        # Provider Selection
        provider_frame = ttk.Frame(model_frame)
        provider_frame.pack(fill=tk.X, pady=5)

        self.provider_label = ttk.Label(provider_frame, text="Provider:", width=20)
        self.provider_label.pack(side=tk.LEFT)

        self.provider_var = tk.StringVar()
        self.provider_combo = ttk.Combobox(
            provider_frame,
            textvariable=self.provider_var,
            values=list(self.models.keys()),
            state="readonly",
            width=30,
        )
        self.provider_combo.pack(side=tk.LEFT, padx=5)
        self.provider_combo.bind("<<ComboboxSelected>>", self._on_provider_change)

        # Model Selection
        model_select_frame = ttk.Frame(model_frame)
        model_select_frame.pack(fill=tk.X, pady=5)

        self.model_label = ttk.Label(model_select_frame, text="Model:", width=20)
        self.model_label.pack(side=tk.LEFT)

        self.model_var = tk.StringVar()
        self.model_combo = ttk.Combobox(
            model_select_frame, textvariable=self.model_var, state="readonly", width=30
        )
        self.model_combo.pack(side=tk.LEFT, padx=5)

        # Save Configuration Button
        self.save_config_button = ttk.Button(
            model_frame, text="Save Configuration", command=self._save_config
        )
        self.save_config_button.pack(pady=5)

        # Middle section - File Selection
        file_frame = ttk.LabelFrame(scrollable_frame, text="File Selection", padding="10")
        file_frame.pack(fill=tk.X, pady=(0, 10))

        # TOC File Selection
        toc_frame = ttk.Frame(file_frame)
        toc_frame.pack(fill=tk.X, pady=5)

        self.toc_label = ttk.Label(toc_frame, text="Table of Contents:", width=20)
        self.toc_label.pack(side=tk.LEFT)

        self.toc_entry = ttk.Entry(toc_frame, textvariable=self.toc_path_var, width=50)
        self.toc_entry.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)

        self.toc_button = ttk.Button(
            toc_frame, text="Browse", command=self._select_toc_file
        )
        self.toc_button.pack(side=tk.LEFT)

        # New: Summary File Selection
        summary_frame = ttk.Frame(file_frame)
        summary_frame.pack(fill=tk.X, pady=5)

        self.summary_label = ttk.Label(summary_frame, text="Summary File:", width=20)
        self.summary_label.pack(side=tk.LEFT)

        self.summary_entry = ttk.Entry(summary_frame, textvariable=self.summary_path_var, width=50)
        self.summary_entry.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)

        self.summary_button = ttk.Button(
            summary_frame, text="Browse", command=self._select_summary_file
        )
        self.summary_button.pack(side=tk.LEFT)

        # Auto-detect button for summary file
        self.auto_detect_button = ttk.Button(
            summary_frame, text="Auto-detect", command=self._auto_detect_summary
        )
        self.auto_detect_button.pack(side=tk.LEFT, padx=5)

        # Output Directory Selection
        output_frame = ttk.Frame(file_frame)
        output_frame.pack(fill=tk.X, pady=5)

        self.output_label = ttk.Label(output_frame, text="Output Directory:", width=20)
        self.output_label.pack(side=tk.LEFT)

        self.output_entry = ttk.Entry(
            output_frame, textvariable=self.output_path_var, width=50
        )
        self.output_entry.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)

        self.output_button = ttk.Button(
            output_frame, text="Browse", command=self._select_output_dir
        )
        self.output_button.pack(side=tk.LEFT)

        # Bottom section - Progress and Logs
        bottom_frame = ttk.Frame(scrollable_frame)
        bottom_frame.pack(fill=tk.BOTH, expand=True)

        # Progress bar
        self.progress_bar = ttk.Progressbar(
            bottom_frame, variable=self.progress_var, maximum=100
        )
        self.progress_bar.pack(fill=tk.X, pady=(0, 10))

        # Log display
        log_frame = ttk.LabelFrame(bottom_frame, text="Logs", padding="10")
        log_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))

        self.log_display = scrolledtext.ScrolledText(log_frame, height=10)
        self.log_display.pack(fill=tk.BOTH, expand=True)

        # Create a separate frame for the start button at the bottom of the main window
        button_frame = ttk.Frame(self.root)
        button_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=10)

        self.start_button = ttk.Button(
            button_frame,
            text="Start Generation",
            command=self.start_generation,
            state=tk.DISABLED,
        )
        self.start_button.pack(pady=5)

        logger.debug("GUI widgets created successfully")

    def _load_config(self) -> None:
        """Load configuration from config.yaml."""
        try:
            with open("config.yaml") as f:
                config = yaml.safe_load(f)
                provider = config.get("provider", "").title()
                logger.debug(f"Loaded provider from config: {provider}")
                
                if provider in self.models:
                    self.provider_var.set(provider)
                    self._update_model_list()

                    # Set the current model from config
                    if provider == "Anthropic":
                        current_model = config.get("anthropic", {}).get("model")
                    else:  # Lingyiwanwu
                        current_model = config.get("lingyiwanwu", {}).get("model")
                    
                    logger.debug(f"Loaded model from config: {current_model}")
                    
                    if current_model in self.models[provider]:
                        self.model_var.set(current_model)
                        logger.info(f"Using {provider} model: {current_model}")
                    else:
                        logger.warning(f"Configured model {current_model} not found in available models")
                        # Set default model
                        self.model_var.set(self.models[provider][0])
                        logger.info(f"Using default model: {self.models[provider][0]}")

                self.update_log("Configuration loaded successfully")
                
        except Exception as e:
            error_msg = f"Error loading configuration: {e}"
            logger.error(error_msg, exc_info=True)
            self.update_log(error_msg)

    def _save_config(self) -> None:
        """Save current configuration to config.yaml."""
        try:
            provider = self.provider_var.get().lower()
            model = self.model_var.get()

            config = {
                "provider": provider,
                "anthropic": {
                    "api_key": "your-anthropic-api-key-here",
                    "model": model
                    if provider == "anthropic"
                    else "claude-3-sonnet-20240229",
                },
                "lingyiwanwu": {
                    "api_key": "your-lingyiwanwu-api-key-here",
                    "model": model if provider == "lingyiwanwu" else "yi-large",
                },
            }

            # Preserve existing API keys
            try:
                with open("config.yaml") as f:
                    old_config = yaml.safe_load(f)
                    if old_config:
                        config["anthropic"]["api_key"] = old_config.get(
                            "anthropic", {}
                        ).get("api_key", "")
                        config["lingyiwanwu"]["api_key"] = old_config.get(
                            "lingyiwanwu", {}
                        ).get("api_key", "")
            except Exception:
                pass

            with open("config.yaml", "w") as f:
                yaml.dump(config, f, default_flow_style=False)

            self.update_log("Configuration saved successfully")
        except Exception as e:
            self.update_log(f"Error saving configuration: {e}")

    def _on_provider_change(self, event=None) -> None:
        """Handle provider selection change."""
        self._update_model_list()

    def _update_model_list(self) -> None:
        """Update model list based on selected provider."""
        provider = self.provider_var.get()
        if provider in self.models:
            self.model_combo["values"] = self.models[provider]
            self.model_combo.set(
                self.models[provider][0]
            )  # Select first model by default

    def _select_toc_file(self) -> None:
        """Open file dialog to select table of contents file."""
        file_path = filedialog.askopenfilename(
            title="Select Table of Contents File",
            filetypes=[("Markdown files", "*.md"), ("All files", "*.*")],
        )
        if file_path:
            self.toc_path = Path(file_path)
            self.toc_path_var.set(str(self.toc_path))
            logger.info(f"Selected TOC file: {self.toc_path}")
            
            # Auto-detect summary file
            summary_path = self.toc_path.parent / f"{self.toc_path.stem}_summary.md"
            if summary_path.exists():
                self.summary_path = summary_path
                self.summary_path_var.set(str(summary_path))
                logger.info(f"Auto-detected summary file: {summary_path}")
            else:
                logger.warning(f"Summary file not found: {summary_path}")
            
            self._validate_inputs()
            self.update_log(f"Selected table of contents: {self.toc_path}")

    def _select_output_dir(self) -> None:
        """Open directory dialog to select output directory."""
        dir_path = filedialog.askdirectory(title="Select Output Directory")
        if dir_path:
            self.output_path = Path(dir_path)
            self.output_path_var.set(str(self.output_path))
            self._validate_inputs()
            self.update_log(f"Selected output directory: {self.output_path}")

    def _select_summary_file(self) -> None:
        """Open file dialog to select summary file."""
        file_path = filedialog.askopenfilename(
            title="Select Summary File",
            filetypes=[("Markdown files", "*.md"), ("All files", "*.*")],
        )
        if file_path:
            self.summary_path = Path(file_path)
            self.summary_path_var.set(str(self.summary_path))
            logger.info(f"Selected summary file: {self.summary_path}")
            self._validate_inputs()
            self.update_log(f"Selected summary file: {self.summary_path}")

    def _auto_detect_summary(self) -> None:
        """Attempt to auto-detect the summary file based on TOC file."""
        if self.toc_path:
            expected_summary = self.toc_path.parent / f"{self.toc_path.stem}_summary.md"
            if expected_summary.exists():
                self.summary_path = expected_summary
                self.summary_path_var.set(str(self.summary_path))
                self._validate_inputs()
                self.update_log(f"Auto-detected summary file: {self.summary_path}")
            else:
                self.update_log("Could not auto-detect summary file")
                messagebox.showwarning(
                    "Auto-detect Failed",
                    f"Expected summary file not found:\n{expected_summary}"
                )
        else:
            messagebox.showwarning(
                "Auto-detect Failed",
                "Please select a table of contents file first"
            )

    def _validate_inputs(self) -> None:
        """Validate inputs and enable/disable start button."""
        if self.toc_path and self.output_path and self.summary_path:
            if (self.toc_path.exists() and 
                self.toc_path.suffix == ".md" and 
                self.summary_path.exists() and 
                self.summary_path.suffix == ".md"):
                self.start_button.config(state=tk.NORMAL)
                return
        self.start_button.config(state=tk.DISABLED)

    def start_generation(self) -> None:
        """Start the content generation process."""
        if not all([self.toc_path, self.output_path, self.summary_path]):
            self.update_log("Error: Please select all required files")
            return

        # Disable all input controls during generation
        self._disable_inputs()
        self._save_config()
        self.update_log("Starting content generation process...")

        # Start generation in a separate thread
        thread = threading.Thread(target=self._generation_process)
        thread.daemon = True
        thread.start()

    def _generation_process(self) -> None:
        """Handle the content generation process in a separate thread."""
        try:
            self.update_log("Step 1/3: Initializing...")
            self.generator = ContentGenerator(
                str(self.toc_path), 
                str(self.output_path)
            )

            if not self.generator:
                raise ValueError("Failed to initialize content generator")

            self.generator.parse_toc()
            self.update_log(
                f"Found {len(self.generator.sections)} sections to process"
            )

            self.update_log("Step 2/3: Starting content generation...")
            self.generator.generate_all_content(self.update_progress)
            self.update_log("Step 3/3: Content generation completed!")

        except Exception as e:
            error_msg = f"Error occurred: {str(e)}"
            self.update_log(error_msg)
            logger.error(f"Error in GUI: {e}")
            messagebox.showerror("Error", error_msg)

        finally:
            self.root.after(0, self._enable_inputs)

    def _enable_inputs(self) -> None:
        """Re-enable input controls."""
        self.start_button.config(state=tk.NORMAL)
        self.toc_button.config(state=tk.NORMAL)
        self.summary_button.config(state=tk.NORMAL)
        self.auto_detect_button.config(state=tk.NORMAL)
        self.output_button.config(state=tk.NORMAL)
        self.provider_combo.config(state="readonly")
        self.model_combo.config(state="readonly")
        self.save_config_button.config(state=tk.NORMAL)

    def update_progress(self, value: float) -> None:
        """Update the progress bar."""
        self.root.after(0, lambda: self.progress_var.set(value))
        self.root.after(0, lambda: self.update_log(f"Progress: {value:.1f}%"))
        self.root.after(0, lambda: self.root.update_idletasks())

    def update_log(self, message: str) -> None:
        """Update the log display."""
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        formatted_message = f"[{timestamp}] {message}"

        def update():
            self.log_display.insert(tk.END, f"{formatted_message}\n")
            self.log_display.see(tk.END)

        self.root.after(0, update)
        logger.info(message)

    def run(self) -> None:
        """Start the GUI main loop."""
        self.root.mainloop()

    def _generate_book_plan(self) -> None:
        """Generate book plan from simple topic input."""
        try:
            topic = self.topic_var.get().strip()
            level = self.level_var.get()
            
            logger.info(f"Starting book plan generation for topic: '{topic}' at {level} level")
            
            if not self._validate_topic_input(topic):
                logger.warning(f"Topic validation failed for: '{topic}'")
                return
            
            self._disable_inputs()
            self.update_log(f"Generating book plan for {topic} at {level} level...")
            
            # Use output_path if set, otherwise use default
            output_dir = self.output_path or self.default_output_dir
            logger.debug(f"Using output directory: {output_dir}")
            
            # Initialize generator
            self.generator = ContentGenerator(
                toc_file="temp_toc.md",
                output_dir=output_dir
            )
            logger.info("ContentGenerator initialized successfully")
            
            # Generate and save content
            try:
                prompt = self.generator._create_book_planning_prompt(
                    f"Create a {level} level tutorial about {topic}"
                )
                content = self.generator._generate_with_retry(prompt)
                toc_content, summary_content = self.generator._parse_ai_response(content)
                
                # Save files
                toc_file = output_dir / "table_of_contents.md"
                summary_file = output_dir / "book_summary.md"
                
                # Create backups if files exist
                for file_path in [toc_file, summary_file]:
                    if file_path.exists():
                        backup_path = file_path.with_suffix(f".bak_{int(time.time())}")
                        file_path.rename(backup_path)
                        logger.info(f"Created backup: {backup_path}")
                
                # Write new files
                toc_file.write_text(toc_content)
                summary_file.write_text(summary_content)
                
                # Update GUI paths
                self.toc_path = toc_file
                self.summary_path = summary_file
                self.toc_path_var.set(str(toc_file))
                self.summary_path_var.set(str(summary_file))
                
                self.update_log("Book plan generated successfully!")
                self.update_log(f"Files saved to: {output_dir}")
                
                self._validate_inputs()
                logger.info("Book plan generation completed successfully")
                
            except Exception as e:
                logger.error(f"Failed to generate or save content: {e}", exc_info=True)
                raise
                
        except Exception as e:
            error_msg = f"Failed to generate book plan: {str(e)}"
            logger.error(f"Book plan generation failed: {e}", exc_info=True)
            self.update_log(f"Error: {error_msg}")
            messagebox.showerror("Error", error_msg)
        finally:
            logger.debug("Re-enabling GUI inputs")
            self._enable_inputs()

    def _disable_inputs(self) -> None:
        """Disable all input widgets during processing."""
        self.topic_entry.config(state="disabled")
        self.level_combo.config(state="disabled")
        self.model_combo.config(state="disabled")
        self.generate_plan_button.config(state="disabled")
        self.toc_button.config(state="disabled")
        self.output_button.config(state="disabled")
        self.start_button.config(state="disabled")

    def _enable_inputs(self) -> None:
        """Enable all input widgets after processing."""
        self.topic_entry.config(state="normal")
        self.level_combo.config(state="readonly")
        self.model_combo.config(state="readonly")
        self.generate_plan_button.config(state="normal")
        self.toc_button.config(state="normal")
        self.output_button.config(state="normal")
        self.start_button.config(state="normal")

    def _validate_topic_input(self, topic: str) -> bool:
        """Validate the topic input string.
        
        Args:
            topic (str): The topic string to validate
            
        Returns:
            bool: True if valid, False otherwise
            
        Note:
            Validates for minimum length, maximum length, and meaningful content
        """
        if not topic or len(topic.strip()) < 3:
            messagebox.showwarning(
                "Invalid Input",
                "Topic must be at least 3 characters long"
            )
            logger.warning("Topic validation failed: Too short")
            return False
            
        if len(topic) > 500:
            messagebox.showwarning(
                "Invalid Input",
                "Topic is too long (max 500 characters)"
            )
            logger.warning("Topic validation failed: Too long")
            return False
            
        if not re.search(r'[a-zA-Z]', topic):
            messagebox.showwarning(
                "Invalid Input",
                "Topic must contain some text characters"
            )
            logger.warning("Topic validation failed: No text characters")
            return False
            
        return True


class ContentGenerationError(Exception):
    """Exception raised for errors in the content generation process.

    Attributes:
        message -- explanation of the error
    """

    def __init__(self, message: str):
        self.message = message
        super().__init__(self.message)


def main():
    """Main entry point for the application."""
    try:
        app = ContentGeneratorGUI()
        app.run()
    except Exception as e:
        logger.error(f"Application error: {e}")
        raise


if __name__ == "__main__":
    main()
