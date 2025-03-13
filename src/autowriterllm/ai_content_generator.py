import logging
from logging.handlers import RotatingFileHandler
import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Callable, Any
import tkinter as tk
from tkinter import ttk, scrolledtext, filedialog, messagebox
import yaml
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
import httpx

# Import from other modules
from autowriterllm.models import Section, ChapterSummary, GenerationProgress, RateLimitConfig, ConceptNode
from autowriterllm.api_client import APIRateLimiter, APIClientFactory
from autowriterllm.cache import ContentCache
from autowriterllm.review import ContentReviewer, ReviewFeedback
from autowriterllm.interactive import InteractiveElementsGenerator, QuizQuestion, Exercise
from autowriterllm.code_quality import CodeQualityAnalyzer, CodeAnalysis
from autowriterllm.learning_path import LearningPathOptimizer, LearningObjective
from autowriterllm.context_management import ContextManager, SemanticContextSearch
from autowriterllm.exceptions import ContentGenerationError

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


class KnowledgeGraph:
    """Maintains relationships between concepts across chapters."""
    def __init__(self):
        self.concepts: Dict[str, ConceptNode] = {}
        
    def add_concept(self, concept: ConceptNode) -> None:
        self.concepts[concept.name] = concept
        
    def get_related_concepts(self, concept_name: str) -> List[str]:
        return self.concepts[concept_name].related_concepts if concept_name in self.concepts else []


class ContentGenerator:
    """Generate content for a book based on a table of contents.

    This class handles the generation of content for a book based on a table of contents.
    It uses AI to generate content for each section.

    Attributes:
        toc_file (Path): Path to the table of contents file
        output_dir (Path): Path to the output directory
        sections (List[Section]): List of sections in the table of contents
        client (Any): API client for content generation
        config (Dict): Configuration dictionary
        chapter_summaries (Dict[str, ChapterSummary]): Dictionary of chapter summaries
    """

    def __init__(self, toc_file: Union[str, Path], output_dir: Union[str, Path]):
        """Initialize the content generator.

        Args:
            toc_file: Path to the table of contents file
            output_dir: Path to the output directory
        """
        self.toc_file = Path(toc_file) if toc_file else None
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.sections: List[Section] = []
        self.chapter_summaries: Dict[str, ChapterSummary] = {}
        self.progress: Optional[GenerationProgress] = None
        self.progress_window: Optional[tk.Toplevel] = None
        self.current_model: Optional[str] = None
        
        # Create summaries directory
        self.summaries_dir = self.output_dir / "summaries"
        self.summaries_dir.mkdir(exist_ok=True, parents=True)

        # Load configuration
        self.config = self._load_config()
        
        # Set current model from config
        provider_name = self.config.get("provider")
        provider_config = self.config.get("providers", {}).get(provider_name, {})
        models = provider_config.get("models", [])
        if models:
            self.current_model = models[0].get("name")
        
        # Initialize API client factory
        self.client_factory = APIClientFactory(self.config)
        self.client = self.client_factory.create_client()
        
        # Initialize cache if directory exists
        cache_dir = self.config.get("cache_dir")
        if cache_dir:
            cache_path = Path(cache_dir)
            cache_path.mkdir(exist_ok=True, parents=True)
            self.cache = ContentCache(cache_path)
            
        # Initialize rate limiter
        rate_limit_config = self.config.get("rate_limit", {})
        if rate_limit_config.get("enabled", False):
            self.rate_limiter = APIRateLimiter(
                RateLimitConfig(
                    max_requests_per_minute=rate_limit_config.get("requests_per_minute", 50)
                )
            )
            
        # Initialize vector database for context
        vector_db_config = self.config.get("vector_db", {})
        if vector_db_config.get("enabled", False):
            self.context_manager = ContextManager(
                collection_name=vector_db_config.get("collection_name", "book_content"),
                embedding_model=vector_db_config.get("embedding_model", "all-MiniLM-L6-v2")
            )
            
        # Initialize knowledge graph
        self.knowledge_graph = KnowledgeGraph()

        # Initialize new components
        self.reviewer = ContentReviewer(self.client)
        self.interactive_generator = InteractiveElementsGenerator(self.client)
        self.code_analyzer = CodeQualityAnalyzer(self.client)
        self.learning_path = LearningPathOptimizer(self.knowledge_graph)
        
    def set_model(self, provider_name: str, model_name: str) -> None:
        """Set the current model for content generation.
        
        Args:
            provider_name: The provider name
            model_name: The model name
            
        Raises:
            ValueError: If the provider or model is not found in the configuration
        """
        providers_config = self.config.get("providers", {})
        if provider_name not in providers_config:
            raise ValueError(f"Provider '{provider_name}' not found in configuration")
            
        provider_config = providers_config[provider_name]
        model_found = False
        
        for model in provider_config.get("models", []):
            if model.get("name") == model_name:
                model_found = True
                break
                
        if not model_found:
            raise ValueError(f"Model '{model_name}' not found for provider '{provider_name}'")
            
        # Update current provider and model
        self.config["provider"] = provider_name
        self.current_model = model_name
        
        # Reinitialize client with new provider
        self.client = self.client_factory.create_client(provider_name)
        logger.info(f"Set model to {provider_name}/{model_name}")

    def _load_config(self) -> Dict:
        """Load configuration from config file.

        Returns:
            Dict: Configuration dictionary

        Raises:
            ValueError: If configuration is invalid
        """
        try:
            # Default configuration
            default_config = {
                "provider": "anthropic",
                "providers": {
                    "anthropic": {
                        "base_url": "https://api.anthropic.com",
                        "models": [
                            {
                                "name": "claude-3-opus-20240229",
                                "description": "Most powerful Claude model",
                                "max_tokens": 8192,
                                "temperature": 0.7
                            },
                            {
                                "name": "claude-3-sonnet-20240229",
                                "description": "Balanced Claude model",
                                "max_tokens": 8192,
                                "temperature": 0.7
                            }
                        ]
                    },
                    "lingyiwanwu": {
                        "base_url": "https://api.lingyiwanwu.com/v1",
                        "models": [
                            {
                                "name": "yi-large",
                                "description": "Large model with enhanced capabilities",
                                "max_tokens": 8192,
                                "temperature": 0.7
                            }
                        ]
                    }
                },
                # Content generation options
                "include_learning_metadata": True,
                "include_previous_chapter_context": True,
                "include_previous_section_context": True,
                "multi_part_generation": True,
                "parts_per_section": 5,
                "max_tokens_per_part": 8192
            }
            
            config_path = Path("config.yaml")
            if config_path.exists():
                with open(config_path) as f:
                    user_config = yaml.safe_load(f)
                    
                    # Validate provider
                    provider = user_config.get("provider")
                    if provider and provider not in user_config.get("providers", {}):
                        logger.warning(f"Provider '{provider}' not found in providers configuration, using default")
                    
                    # Merge user config with default config
                    for key, value in user_config.items():
                        if isinstance(value, dict) and key in default_config and isinstance(default_config[key], dict):
                            # Deep merge for nested dictionaries
                            self._deep_merge_dict(default_config[key], value)
                        else:
                            default_config[key] = value
                    
                    return default_config
            else:
                logger.warning("Config file not found, using default configuration")
                return default_config
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            raise

    def _deep_merge_dict(self, target: Dict, source: Dict) -> None:
        """Deep merge two dictionaries.
        
        Args:
            target: Target dictionary to merge into
            source: Source dictionary to merge from
        """
        for key, value in source.items():
            if key in target and isinstance(target[key], dict) and isinstance(value, dict):
                self._deep_merge_dict(target[key], value)
            else:
                target[key] = value

    def _read_file_with_encoding(self, file_path: Path) -> str:
        """Read file content with multiple encoding attempts.
        
        Args:
            file_path: Path to the file to read
            
        Returns:
            str: File content
            
        Raises:
            UnicodeDecodeError: If file cannot be read with any encoding
        """
        encodings = ['utf-8', 'gbk', 'gb2312', 'gb18030', 'big5', 'utf-16']
        
        for encoding in encodings:
            try:
                content = file_path.read_text(encoding=encoding)
                logger.debug(f"Successfully read file {file_path} with {encoding} encoding")
                return content
            except UnicodeDecodeError:
                continue
                
        raise UnicodeDecodeError(
            f"Failed to read {file_path} with any of these encodings: {encodings}"
        )

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
        try:
            if not self.toc_file.exists():
                logger.error(f"TOC file does not exist: {self.toc_file}")
                raise FileNotFoundError(f"TOC file not found: {self.toc_file}")

            content = self._read_file_with_encoding(self.toc_file)
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

    def _load_existing_summaries(self) -> None:
        """Load previously generated chapter summaries."""
        try:
            for summary_file in self.summaries_dir.glob("*.yaml"):
                with open(summary_file) as f:
                    data = yaml.safe_load(f)
                    self.chapter_summaries[data["title"]] = ChapterSummary(**data)
                    logger.debug(f"Loaded existing summary for: {data['title']}")
        except Exception as e:
            logger.error(f"Error loading existing summaries: {e}")

    def _save_chapter_summary(self, title: str, summary: str) -> None:
        """Save chapter summary to disk."""
        try:
            summary_obj = ChapterSummary(title=title, summary=summary)
            self.chapter_summaries[title] = summary_obj
            
            # Save to file
            summary_file = self.summaries_dir / f"{hashlib.md5(title.encode()).hexdigest()}.yaml"
            with open(summary_file, 'w') as f:
                yaml.dump(summary_obj.__dict__, f)
            logger.debug(f"Saved summary for chapter: {title}")
        except Exception as e:
            logger.error(f"Error saving chapter summary: {e}")

    def _get_chapter_summaries_context(self, current_section: Section) -> str:
        """Get formatted context from all previous chapter summaries.
        
        Args:
            current_section: The current section being generated
            
        Returns:
            str: Formatted chapter summaries context
        """
        context_parts = []
        
        try:
            # Find current chapter index
            current_index = next(
                (i for i, s in enumerate(self.sections) if s.title == current_section.title), 
                -1
            )
            
            # Get summaries for all previous chapters
            for section in self.sections[:current_index]:
                if summary := self.chapter_summaries.get(section.title):
                    context_parts.append(f"\nChapter: {section.title}\nSummary: {summary.summary}")
                else:
                    # If no summary exists, try to generate one from the content file
                    content_file = self.output_dir / self._create_filename(section)
                    if content_file.exists():
                        content = content_file.read_text(encoding='utf-8')
                        summary = self._request_chapter_summary(section, content)
                        context_parts.append(f"\nChapter: {section.title}\nSummary: {summary}")
            
            return "\n".join(context_parts)
            
        except Exception as e:
            logger.error(f"Error getting chapter summaries context: {e}")
            return "No previous chapter summaries available"

    def _request_chapter_summary(self, section: Section, content: str) -> str:
        """Request AI to generate a chapter summary.
        
        This generates a concise summary of the chapter's key concepts without
        recapping previous chapters. The summary is used for internal context
        management only and is not included in the final content.
        
        Args:
            section: The section to summarize
            content: The content to summarize
            
        Returns:
            str: The generated summary
        """
        try:
            summary_prompt = f"""Please provide a concise summary (2-3 sentences) of the following chapter content. 
Focus ONLY on the key concepts covered in THIS chapter.
DO NOT reference or recap previous chapters.
DO NOT include introductory phrases like "This chapter covers..." or "In this chapter, we learned..."

Chapter: {section.title}

Content:
{content[:2000]}  # Using first 2000 chars for summary

Generate a summary that captures the essential concepts of this chapter only."""

            summary = self._generate_with_retry(summary_prompt)
            return summary.strip()
        except Exception as e:
            logger.error(f"Error generating chapter summary: {e}")
            return ""

    def generate_section_content(self, section: Section) -> Tuple[str, str]:
        """Generate content for a section.
        
        Args:
            section: The section to generate content for
            
        Returns:
            Tuple[str, str]: The generated content and filename
        """
        content_type = self._determine_tutorial_type()
        
        # Check if content is already cached
        cache_key = None
        if hasattr(self, 'cache'):
            prompt = self._create_prompt(section)
            cache_key = self.cache.get_cache_key(section, prompt)
            cached_content = self.cache.get_cached_content(cache_key)
            if cached_content:
                logger.info(f"Using cached content for {section.title}")
                filename = self._create_filename(section)
                return cached_content, filename
        
        # Determine if this is an introduction
        is_introduction = section.title.lower().startswith("introduction") or section.level == 1
        
        # Generate content in multiple parts for more comprehensive coverage
        try:
            # Check if multi-part generation is enabled
            use_multi_part = self.config.get("multi_part_generation", True)
            
            if use_multi_part:
                # Number of parts to generate for each section
                total_parts = self.config.get("parts_per_section", 5)
                all_content_parts = []
                
                logger.info(f"Generating content for '{section.title}' in {total_parts} parts")
                
                for part_number in range(1, total_parts + 1):
                    logger.info(f"Generating part {part_number}/{total_parts} for '{section.title}'")
                    
                    # Wait for rate limiting if needed
                    if hasattr(self, 'rate_limiter'):
                        self.rate_limiter.wait_if_needed()
                    
                    # Create prompt with context for this specific part
                    prompt = self._create_prompt(section, is_introduction, part_number=part_number, total_parts=total_parts)
                    
                    # Generate content for this part
                    part_content = self._generate_with_retry(prompt)
                    all_content_parts.append(part_content)
                    
                    logger.info(f"Successfully generated part {part_number}/{total_parts} for '{section.title}'")
                
                # Combine all parts into a single content
                content = "\n\n".join(all_content_parts)
            else:
                # Generate content in a single part (original method)
                logger.info(f"Generating content for '{section.title}' in a single part")
                
                # Wait for rate limiting if needed
                if hasattr(self, 'rate_limiter'):
                    self.rate_limiter.wait_if_needed()
                
                # Create prompt with context
                prompt = self._create_prompt(section, is_introduction)
                
                # Generate content
                content = self._generate_with_retry(prompt)
                
                logger.info(f"Successfully generated content for '{section.title}'")
            
            # Improve content quality
            if hasattr(self, 'code_analyzer') and content_type == "programming":
                analysis = self.code_analyzer.analyze_code(content)
                content = self._improve_code_sections(content, analysis)
            
            if hasattr(self, 'content_analyzer'):
                analysis = self.content_analyzer.analyze_content(content, content_type)
                # Implement content improvement based on analysis
            
            if hasattr(self, 'reviewer'):
                review = self.reviewer.review_content(content, section.title, content_type)
                content = self._improve_content_based_on_review(content, review, content_type)
            
            # Add interactive elements if appropriate
            if hasattr(self, 'interactive_generator'):
                if "exercise" in section.title.lower() or "practice" in section.title.lower():
                    exercises = self.interactive_generator.generate_exercises(content, "intermediate", content_type)
                    quiz_questions = self.interactive_generator.generate_quiz(content, "medium", content_type)
                    content = self._add_interactive_elements(content, quiz_questions, exercises)
            
            # Add learning objectives if appropriate
            if hasattr(self, 'learning_path_optimizer'):
                objectives = self.learning_path_optimizer.generate_learning_objectives(section, content_type)
                prerequisites = self.learning_path_optimizer.suggest_prerequisites(section, content_type)
                content = self._add_learning_metadata(content, objectives, prerequisites)
            
            # Update knowledge graph
            self._update_knowledge_graph(section, content)
            
            # Generate chapter summary
            if not section.parent:  # If it's a main chapter
                summary = self._request_chapter_summary(section, content)
                self._save_chapter_summary(section.title, summary)
            
            # Cache the content
            if hasattr(self, 'cache') and cache_key:
                self.cache.cache_content(cache_key, content)
            
            # Create filename
            filename = self._create_filename(section)
            
            # Add to vector database for context
            self._add_to_vector_db(filename, content)
            
            return content, filename
        except Exception as e:
            logger.error(f"Error generating content for {section.title}: {e}")
            raise ContentGenerationError(f"Failed to generate content for {section.title}: {str(e)}")

    def _load_all_previous_content(self, current_section: Section) -> Dict[str, str]:
        """Load content from all previous sections.
        
        Args:
            current_section: The current section being generated
            
        Returns:
            Dict mapping section titles to their content
        """
        previous_content = {}
        
        try:
            # Find current section index
            current_index = next(
                (i for i, s in enumerate(self.sections) if s.title == current_section.title), 
                -1
            )
            
            # Load content from all previous sections
            for section in self.sections[:current_index]:
                content_file = self.output_dir / self._create_filename(section)
                if content_file.exists():
                    try:
                        content = content_file.read_text(encoding='utf-8')
                        previous_content[section.title] = content
                        logger.debug(f"Loaded content from: {section.title}")
                    except Exception as e:
                        logger.warning(f"Could not read content for {section.title}: {e}")
            
            return previous_content
            
        except Exception as e:
            logger.error(f"Error loading previous content: {e}")
            return {}

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
            client = chromadb.Client(Settings(
                is_persistent=True,
                persist_directory=str(db_path),
                anonymized_telemetry=False,
            ))

            collection_name = "sections"
            
            try:
                collection = client.get_collection(collection_name)
                logger.info(f"Using existing collection: {collection_name}")
            except Exception:
                logger.info(f"Creating new collection: {collection_name}")
                collection = client.create_collection(
                    name=collection_name,
                    metadata={"hnsw:space": "cosine"}  # Use cosine similarity
                )

            # Get current chapter number
            current_chapter = int(section.title.split('.')[0])
            
            # Index content from previous chapters
            indexed_count = 0
            for prev_section in self.sections:
                prev_chapter = int(prev_section.title.split('.')[0])
                if prev_chapter >= current_chapter:
                    break
                
                file_path = self.output_dir / self._create_filename(prev_section)
                if file_path.exists():
                    try:
                        content = file_path.read_text(encoding='utf-8')
                        collection.upsert(
                            ids=[prev_section.title],
                            documents=[content],
                            metadatas=[{
                                "title": prev_section.title,
                                "chapter": prev_chapter
                            }]
                        )
                        indexed_count += 1
                        logger.debug(f"Indexed content for: {prev_section.title}")
                    except Exception as e:
                        logger.error(f"Error indexing {prev_section.title}: {e}")

            # Query with chapter-aware filtering
            if indexed_count > 0:
                results = collection.query(
                    query_texts=[section.title],
                    n_results=min(3, indexed_count),
                    where={"chapter": {"$lt": current_chapter}},
                    include=["documents", "metadatas", "distances"]
                )
                
                if results and results["documents"] and results["documents"][0]:
                    context_parts = []
                    for doc, metadata, distance in zip(
                        results["documents"][0],
                        results["metadatas"][0],
                        results["distances"][0]
                    ):
                        # Extract most relevant part (last 500 chars)
                        relevant_content = doc[-500:] if len(doc) > 500 else doc
                        similarity_score = 1 - distance  # Convert distance to similarity
                        
                        if similarity_score > 0.3:  # Only include if somewhat relevant
                            context = (
                                f"\nContext from {metadata['title']} "
                                f"(similarity: {similarity_score:.2f}):\n{relevant_content}"
                            )
                            context_parts.append(context)
                            logger.debug(
                                f"Added context from {metadata['title']} "
                                f"with similarity {similarity_score:.2f}"
                            )
                    
                    if context_parts:
                        return "\n\n".join(context_parts)
                    else:
                        logger.info("No sufficiently relevant context found")
                        return ""
                        
            else:
                logger.info("First section - no previous context needed")
                return ""

        except Exception as e:
            logger.error(f"Error in vector search: {e}", exc_info=True)
            return ""

    def _determine_tutorial_type(self) -> str:
        """Determine the type of content being generated.
        
        Returns:
            str: The content type (e.g., 'programming', 'science', 'history', etc.)
        """
        try:
            # Try to infer from the table of contents
            toc_content = self._read_file_with_encoding(self.toc_file)
            
            # Look for keywords in the TOC to determine the type
            content_types = {
                'programming': ['code', 'programming', 'function', 'class', 'algorithm', 'development'],
                'science': ['experiment', 'theory', 'hypothesis', 'research', 'scientific'],
                'mathematics': ['theorem', 'proof', 'equation', 'calculation', 'formula'],
                'history': ['century', 'era', 'period', 'historical', 'timeline'],
                'literature': ['novel', 'poetry', 'author', 'literary', 'character'],
                'business': ['management', 'strategy', 'marketing', 'finance', 'economics'],
                'arts': ['painting', 'sculpture', 'composition', 'design', 'artistic'],
                'medicine': ['diagnosis', 'treatment', 'patient', 'clinical', 'medical'],
                'philosophy': ['ethics', 'metaphysics', 'philosophy', 'reasoning', 'logic'],
                'language': ['grammar', 'vocabulary', 'pronunciation', 'linguistic', 'syntax']
            }
            
            # Count occurrences of keywords
            type_scores = {category: 0 for category in content_types}
            for category, keywords in content_types.items():
                for keyword in keywords:
                    type_scores[category] += toc_content.lower().count(keyword.lower())
            
            # Get the category with the highest score
            if any(type_scores.values()):
                content_type = max(type_scores.items(), key=lambda x: x[1])[0]
                logger.info(f"Detected content type: {content_type}")
                return content_type
            
            # Default to generic if no clear type is detected
            return "generic"
        except Exception as e:
            logger.warning(f"Error determining tutorial type: {e}")
            return "generic"

    def _create_prompt(self, section: Section, is_introduction: bool = False, 
                      context: Dict[str, Any] = None, part_number: int = None, total_parts: int = None) -> str:
        """Create a prompt for content generation.
        
        Args:
            section: The section to generate content for
            is_introduction: Whether this is an introduction section
            context: Additional context information
            part_number: Current part number when generating content in multiple parts
            total_parts: Total number of parts for the section
            
        Returns:
            str: The prompt for the AI
        """
        content_type = self._determine_tutorial_type()
        
        # Get context from previous sections
        previous_context = self._get_immediate_previous_context(section)
        
        # Get context from vector database if available
        vector_context = ""
        if hasattr(self, 'context_manager') and self.context_manager:
            vector_context = self._get_context_using_vectors(section)
        
        # Combine contexts
        combined_context = f"{previous_context}\n\n{vector_context}".strip()
        
        # Create specific prompt based on content type
        if is_introduction:
            # Create introduction-specific prompt
            return self._create_specific_prompt(section, combined_context, is_introduction, part_number, total_parts)
        
        # Base prompt structure with multi-part instructions if applicable
        part_instructions = ""
        if part_number is not None and total_parts is not None:
            part_instructions = f"""
IMPORTANT: This is part {part_number} of {total_parts} for this section. 
"""
            if part_number == 1:
                part_instructions += """
For this first part:
- Provide a brief introduction to the topic
- Begin explaining the core concepts in depth
- Include detailed examples and illustrations
- DO NOT include a summary at the end of this part
"""
            elif part_number < total_parts:
                part_instructions += f"""
For part {part_number}:
- Continue directly from where part {part_number-1} left off
- DO NOT recap what was covered in previous parts
- Focus on new content and examples
- Explore concepts in greater depth
- DO NOT include a summary at the end of this part
"""
            else:  # Last part
                part_instructions += f"""
For this final part:
- Continue directly from where part {part_number-1} left off
- DO NOT recap what was covered in previous parts
- Complete any remaining explanations and examples
- Provide a concise conclusion for the entire section
"""
        
        prompt = f"""You are an expert author writing a comprehensive textbook on {content_type}.
Write the content for the section titled "{section.title}".{part_instructions}

CONTEXT INFORMATION:
{combined_context}

GUIDELINES:
1. Write detailed, accurate, and educational content
2. Use clear explanations with MULTIPLE in-depth examples
3. Include relevant diagrams, tables, or illustrations (described in markdown)
4. Add practical examples where appropriate
5. Format using proper markdown with headings, lists, and emphasis
6. Ensure content is engaging and accessible to the target audience
7. DO NOT recap content from previous chapters
8. DO NOT include unnecessary summaries within the content
9. Focus on providing comprehensive, in-depth explanations
"""

        # Add content-specific guidelines based on the type
        if content_type == "programming":
            prompt += """
10. Include well-commented code examples
11. Explain algorithms and data structures clearly
12. Cover best practices and common pitfalls
13. Include debugging tips and performance considerations
14. Provide multiple code examples for each concept
"""
        elif content_type == "science":
            prompt += """
10. Include scientific principles and theories
11. Explain experimental methods and results
12. Reference important studies and findings
13. Include diagrams of scientific processes
14. Provide multiple examples of scientific applications
"""
        elif content_type == "mathematics":
            prompt += """
10. Include clear mathematical notation using LaTeX
11. Provide step-by-step derivations of formulas
12. Include visual representations of mathematical concepts
13. Provide practice problems with solutions
14. Offer multiple approaches to solving problems
"""
        elif content_type == "history":
            prompt += """
10. Include accurate dates and chronology
11. Provide context for historical events
12. Include perspectives from different historical sources
13. Connect historical events to broader themes
14. Analyze historical events in depth
"""
        elif content_type == "literature":
            prompt += """
10. Include analysis of literary techniques
11. Provide examples from important texts
12. Discuss cultural and historical context of works
13. Include biographical information about key authors
14. Offer multiple interpretations of literary works
"""
        elif content_type == "business":
            prompt += """
10. Include case studies of real businesses
11. Provide practical frameworks and models
12. Include market analysis and trends
13. Discuss ethical considerations in business
14. Offer multiple business strategy examples
"""
        elif content_type == "arts":
            prompt += """
10. Include descriptions of artistic techniques
11. Provide historical context for artistic movements
12. Include analysis of notable works
13. Discuss the cultural impact of art
14. Offer multiple examples of artistic styles
"""
        elif content_type == "medicine":
            prompt += """
10. Include accurate medical terminology
11. Describe diagnostic criteria and treatment protocols
12. Include anatomical diagrams and illustrations
13. Discuss current research and evidence-based practices
14. Provide multiple clinical examples
"""
        elif content_type == "philosophy":
            prompt += """
10. Include clear explanations of philosophical concepts
11. Provide historical context for philosophical ideas
12. Present different perspectives on philosophical questions
13. Include thought experiments and logical arguments
14. Analyze philosophical arguments in depth
"""
        elif content_type == "language":
            prompt += """
10. Include examples of language usage
11. Provide clear explanations of grammatical rules
12. Include pronunciation guides where relevant
13. Discuss cultural aspects of language
14. Offer multiple examples of language patterns
"""
        
        # Add section-specific instructions
        if "exercise" in section.title.lower() or "practice" in section.title.lower():
            prompt += """
ADDITIONAL INSTRUCTIONS:
- Create practical exercises for readers to apply what they've learned
- Include a mix of easy, medium, and challenging exercises
- Provide hints for difficult exercises
- Include answer keys or solution guidelines
- Create a large number of varied exercises
"""
        
        if "summary" in section.title.lower() or "conclusion" in section.title.lower():
            prompt += """
ADDITIONAL INSTRUCTIONS:
- Summarize the key points covered in the chapter
- Reinforce the most important concepts
- Connect the material to the broader subject
- Suggest further reading or next steps
"""
        
        if "case study" in section.title.lower() or "example" in section.title.lower():
            prompt += """
ADDITIONAL INSTRUCTIONS:
- Provide a detailed, realistic case study
- Connect theoretical concepts to practical application
- Include analysis and lessons learned
- Suggest how readers can apply similar approaches
- Provide multiple in-depth case studies
"""
        
        prompt += f"""
Write comprehensive content for the section titled "{section.title}".
"""
        
        return prompt

    def _get_immediate_previous_context(self, section: Section) -> str:
        """Get context from immediately preceding section.
        
        This method retrieves context from the immediately preceding section
        if configured to do so. It focuses on key concepts rather than providing
        a full recap of previous content.
        
        Args:
            section: The current section
            
        Returns:
            str: The immediate previous context
        """
        try:
            # Check if we should include previous section context
            if not self.config.get("include_previous_section_context", True):
                return ""
                
            current_index = next(
                (i for i, s in enumerate(self.sections) if s.title == section.title),
                -1
            )
            
            if current_index > 0:
                prev_section = self.sections[current_index - 1]
                prev_file = self.output_dir / self._create_filename(prev_section)
                
                if prev_file.exists():
                    content = self._read_file_with_encoding(prev_file)
                    
                    # If this is a new chapter, only provide minimal context
                    if section.level == 1:
                        # Just provide the title of the previous chapter
                        return f"Previous chapter: {prev_section.title}"
                    
                    # Extract conclusion or summary section if present
                    summary_match = re.search(r'#+\s*(?:Summary|Conclusion).*?\n(.*?)(?:\n#|$)', 
                                            content, re.DOTALL)
                    if summary_match:
                        # Limit the summary to a reasonable length
                        summary = summary_match.group(1).strip()
                        if len(summary) > 500:
                            summary = summary[:500] + "..."
                        return summary
                    
                    # Otherwise return last paragraph, but limit length
                    paragraphs = content.split('\n\n')
                    last_para = paragraphs[-1].strip()
                    if len(last_para) > 300:
                        last_para = last_para[:300] + "..."
                    return last_para
                    
            return ""
            
        except Exception as e:
            logger.error(f"Error getting immediate previous context: {e}")
            return ""

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
                    subsection_file = self.output_dir / self._create_filename(
                        Section(title=subsection, subsections=[], level=3, parent=section.title)
                    )
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
        """Create a prompt for generating a book plan.
        
        Args:
            user_input: User's description of the book they want to create
            
        Returns:
            str: The prompt for the AI
        """
        # Extract key information from user input
        topic_match = re.search(r'(create|generate|write|make)\s+(\w+)\s+(.*?)\s+(book|guide|tutorial|textbook)', 
                              user_input.lower())
        
        level = "intermediate"
        topic = user_input
        
        if topic_match:
            level = topic_match.group(2)
            topic = topic_match.group(3)
        
        prompt = f"""You are an expert author and educator. Create a detailed plan for a {level} level textbook on "{topic}".

TASK:
1. Generate a comprehensive table of contents with chapters and subsections
2. Create a book summary that explains the purpose, target audience, and prerequisites

GUIDELINES:
1. The table of contents should be well-structured and logical
2. Include 5-10 main chapters with 3-5 subsections each
3. Cover fundamentals through advanced topics in a progressive sequence
4. Include practical examples, case studies, and exercises
5. Consider the {level} level of the target audience
6. Make sure the content is comprehensive and educational

FORMAT YOUR RESPONSE EXACTLY AS FOLLOWS:

```markdown:table_of_contents.md
# Book Title
## 1. Chapter Title
### 1.1 Section Title
### 1.2 Section Title
## 2. Chapter Title
...
```

```markdown:book_summary.md
[Write a 3-5 paragraph summary of the book here, including purpose, audience, and prerequisites]
```

Do not include any other text in your response besides these two markdown code blocks.
"""
        
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
        """Generate content with automatic retry on failure.
        
        Args:
            prompt: The prompt to send to the AI
            max_retries: Maximum number of retries on failure
            
        Returns:
            str: The generated content
            
        Raises:
            ContentGenerationError: If content generation fails after all retries
        """
        try:
            # Use the client directly instead of creating a new APIContentGenerator
            provider_name = self.config.get("provider")
            providers_config = self.config.get("providers", {})
            provider_config = providers_config.get(provider_name, {})
            
            # Get the current model configuration
            current_model = None
            model_name = self.current_model
            
            # Find the current model in the provider's models list
            for model in provider_config.get("models", []):
                if model.get("name") == model_name:
                    current_model = model
                    break
            
            # If model not found, use the first model in the list
            if not current_model and provider_config.get("models"):
                current_model = provider_config["models"][0]
                model_name = current_model.get("name")
                logger.warning(f"Model {self.current_model} not found, using {model_name}")
            
            # Get model parameters
            max_tokens = current_model.get("max_tokens") if current_model else self.config.get("max_tokens_per_part", 8192)
            temperature = current_model.get("temperature") if current_model else 0.7
            
            # Apply rate limiting if available
            if hasattr(self, 'rate_limiter'):
                self.rate_limiter.wait_if_needed()
            
            for attempt in range(max_retries):
                try:
                    logger.info(f"Generation attempt {attempt + 1}/{max_retries}")
                    
                    # Use the client directly
                    if provider_name == "anthropic":
                        logger.debug(f"Using Anthropic API with model {model_name}")
                        response = self.client.messages.create(
                            model=model_name,
                            max_tokens=max_tokens,
                            temperature=temperature,
                            messages=[{"role": "user", "content": prompt}]
                        )
                        content = response.content[0].text
                        logger.info("Successfully generated content from Anthropic API")
                        return content
                        
                    elif provider_name == "openai":
                        logger.debug(f"Using OpenAI API with model {model_name}")
                        response = self.client.chat.completions.create(
                            model=model_name,
                            max_tokens=max_tokens,
                            temperature=temperature,
                            messages=[{"role": "user", "content": prompt}]
                        )
                        content = response.choices[0].message.content
                        logger.info("Successfully generated content from OpenAI API")
                        return content
                        
                    else:
                        # Generic API call for other providers
                        logger.debug(f"Using {provider_name} API with model {model_name}")
                        
                        # Try to use the chat completions endpoint (OpenAI-compatible)
                        try:
                            response = self.client.chat.completions.create(
                                model=model_name,
                                max_tokens=max_tokens,
                                temperature=temperature,
                                messages=[{"role": "user", "content": prompt}]
                            )
                            content = response.choices[0].message.content
                            logger.info(f"Successfully generated content from {provider_name} API")
                            return content
                        except (AttributeError, NotImplementedError) as e:
                            # If chat completions not available, try direct HTTP request
                            logger.debug(f"Chat completions not available for {provider_name}, using HTTP request: {e}")
                            
                            url = f"{provider_config.get('base_url')}/chat/completions"
                            headers = {
                                "Authorization": f"Bearer {provider_config.get('api_key')}",
                                "Content-Type": "application/json"
                            }
                            
                            data = {
                                "model": model_name,
                                "messages": [
                                    {"role": "user", "content": prompt}
                                ],
                                "temperature": temperature,
                                "max_tokens": max_tokens
                            }
                            
                            logger.debug(f"Request data: {data}")
                            response = requests.post(url, headers=headers, json=data, timeout=120)
                            
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
                            
                            logger.info(f"Successfully generated content from {provider_name} API")
                            return content
                        
                except (requests.Timeout, requests.ConnectionError) as e:
                    logger.warning(f"Network error on attempt {attempt + 1}: {e}")
                    if attempt == max_retries - 1:
                        raise ContentGenerationError(f"Network error after {max_retries} attempts")
                    wait_time = min(300, 10 * (2 ** attempt))
                    logger.info(f"Waiting {wait_time} seconds before retry...")
                    time.sleep(wait_time)
                    
                except Exception as e:
                    logger.error(f"Error on attempt {attempt + 1}: {e}")
                    if attempt == max_retries - 1:
                        raise ContentGenerationError(f"Failed after {max_retries} attempts: {str(e)}")
                    wait_time = min(300, 10 * (2 ** attempt))
                    logger.info(f"Waiting {wait_time} seconds before retry...")
                    time.sleep(wait_time)
            
            raise ContentGenerationError("Failed to generate content after all retries")
            
        except Exception as e:
            logger.error(f"Error generating content: {e}")
            raise ContentGenerationError(f"Failed to generate content: {str(e)}")

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
        """Safely write content to file with error handling.
        
        Args:
            file_path: Path to write to
            content: Content to write
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Create parent directories if they don't exist
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Create backup if file exists
            if file_path.exists():
                backup_path = file_path.with_suffix(f".bak_{int(time.time())}")
                file_path.rename(backup_path)
                logger.info(f"Created backup: {backup_path}")
            
            # Write new content
            file_path.write_text(content, encoding='utf-8')
            logger.info(f"Successfully wrote file: {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error writing file {file_path}: {e}")
            return False

    def _extract_key_concepts(self, content: str) -> List[ConceptNode]:
        """Extract key concepts from generated content."""
        concept_prompt = f"""Analyze this content and identify key concepts:
{content[:2000]}

For each concept, provide:
1. Concept name
2. Brief description
3. Required prerequisites
4. Related concepts

Format as YAML."""
        
        try:
            concepts_yaml = self._generate_with_retry(concept_prompt)
            concepts_data = yaml.safe_load(concepts_yaml)
            return [ConceptNode(**concept) for concept in concepts_data]
        except Exception as e:
            logger.error(f"Failed to extract concepts: {e}")
            return []

    def _update_knowledge_graph(self, section: Section, content: str) -> None:
        """Update the knowledge graph with concepts from the content.
        
        Args:
            section: The section being processed
            content: The content to extract concepts from
        """
        content_type = self._determine_tutorial_type()
        try:
            concepts = self._extract_key_concepts(content)
            for concept in concepts:
                self.knowledge_graph.add_concept(concept)
        except Exception as e:
            logger.error(f"Failed to update knowledge graph: {e}")

    def _improve_code_sections(self, content: str, analysis: CodeAnalysis) -> str:
        """Improve code sections based on analysis.
        
        Args:
            content: The content containing code sections
            analysis: Code analysis results
            
        Returns:
            str: Improved content
        """
        if not analysis.suggestions:
            return content
        
        # Log improvement suggestions
        logger.info(f"Improving code sections with {len(analysis.suggestions)} suggestions")
        
        # Create a prompt for improving the code
        prompt = f"""Improve the following content based on these suggestions:

CONTENT:
{content[:3000]}

SUGGESTIONS:
{chr(10).join(f'- {suggestion}' for suggestion in analysis.suggestions)}

Return the improved content with the same structure but better code examples.
"""
        
        try:
            improved_content = self._generate_with_retry(prompt)
            return improved_content
        except Exception as e:
            logger.error(f"Failed to improve code sections: {e}")
            return content

    def _improve_content_based_on_review(self, content: str, review: ReviewFeedback, content_type: str = "generic") -> str:
        """Improve content based on review feedback.
        
        Args:
            content: The content to improve
            review: Review feedback
            content_type: Type of content
            
        Returns:
            str: Improved content
        """
        if not review.suggested_improvements:
            return content
        
        # Log improvement suggestions
        logger.info(f"Improving content with {len(review.suggested_improvements)} suggestions")
        
        # Create a prompt for improving the content
        prompt = f"""Improve the following {content_type} textbook content based on these suggestions:

CONTENT:
{content[:3000]}

SUGGESTIONS:
{chr(10).join(f'- {suggestion}' for suggestion in review.suggested_improvements)}

SCORES:
- Technical accuracy: {review.technical_accuracy}
- Clarity: {review.clarity}
- Completeness: {review.completeness}

Return the improved content with the same structure but addressing the suggestions.
"""
        
        try:
            improved_content = self._generate_with_retry(prompt)
            return improved_content
        except Exception as e:
            logger.error(f"Failed to improve content based on review: {e}")
            return content

    def _add_interactive_elements(self, content: str, 
                            quiz_questions: List[QuizQuestion],
                            exercises: List[Exercise]) -> str:
        """Add interactive elements to content."""
        try:
            quiz_section = "\n\n## Quiz Questions\n\n"
            for i, question in enumerate(quiz_questions, 1):
                quiz_section += f"{i}. {question.question}\n"
                for option in question.options:
                    quiz_section += f"   - {option}\n"
                quiz_section += f"\nExplanation: {question.explanation}\n\n"
            
            exercise_section = "\n\n## Practical Exercises\n\n"
            for i, exercise in enumerate(exercises, 1):
                exercise_section += f"### Exercise {i}: {exercise.title}\n\n"
                exercise_section += f"{exercise.description}\n\n"
                exercise_section += "Steps:\n"
                for step in exercise.steps:
                    exercise_section += f"1. {step}\n"
                exercise_section += f"\nExpected Outcome: {exercise.expected_outcome}\n\n"
            
            return content + quiz_section + exercise_section
        except Exception as e:
            logger.error(f"Failed to add interactive elements: {e}")
            return content

    def _add_learning_metadata(self, content: str,
                         objectives: List[LearningObjective],
                         prerequisites: List[str]) -> str:
        """Add learning objectives and prerequisites to content.
        
        This method adds learning objectives and prerequisites to the content
        if configured to do so. The metadata is added at the beginning of the
        content and focuses only on the current chapter without recapping
        previous chapters.
        
        Args:
            content: The content to add metadata to
            objectives: List of learning objectives
            prerequisites: List of prerequisites
            
        Returns:
            str: The content with metadata added
        """
        try:
            # Check if learning metadata should be included
            if not self.config.get("include_learning_metadata", True):
                return content
                
            metadata = "## Learning Objectives\n\n"
            for objective in objectives:
                metadata += f"- {objective.description}\n"
            
            # Only include prerequisites if there are any
            if prerequisites:
                metadata += "\n## Prerequisites\n\n"
                for prereq in prerequisites:
                    metadata += f"- {prereq}\n"
            
            return metadata + "\n\n" + content
        except Exception as e:
            logger.error(f"Failed to add learning metadata: {e}")
            return content

    def _create_hierarchical_context(self, section: Section) -> str:
        """Create a hierarchical context for the current section.
        
        This method creates a context for the current section based on previous
        chapters. It limits the amount of context to avoid overwhelming the
        content generation with too much previous information.
        
        Args:
            section: The current section
            
        Returns:
            str: The hierarchical context
        """
        context_parts = []
        
        # Find current chapter index
        current_index = next(
            (i for i, s in enumerate(self.sections) if s.title == section.title), -1
        )
        
        # Get summaries for only the most recent chapters (limit to 3)
        # This prevents overwhelming the context with too much previous information
        recent_sections = self.sections[max(0, current_index-3):current_index]
        
        # Check if we should include previous chapter context
        if not self.config.get("include_previous_chapter_context", True):
            return ""
        
        for section in recent_sections:
            if summary := self.chapter_summaries.get(section.title):
                context_parts.append(f"\nChapter: {section.title}\nKey Concepts: {summary}")
        
        return "\n".join(context_parts)

    def _combine_context(self, *contexts: str) -> str:
        """Combine multiple context strings intelligently."""
        combined_context = ""
        for context in contexts:
            if context:
                combined_context += context + "\n\n"
        return combined_context.strip()

    def _generate_with_context(self, section: Section, context: str) -> str:
        """Generate content with enhanced context."""
        try:
            prompt = self._create_prompt(section, context=context)
            return self._generate_with_retry(prompt)
        except Exception as e:
            logger.error(f"Failed to generate content with context: {e}")
            raise ContentGenerationError(f"Failed to generate content with context: {e}")

    def _get_parent_context(self, section: Section) -> str:
        """Get comprehensive context from parent section."""
        if not section.parent:
            return ""
        
        try:
            parent_file = self.output_dir / self._create_filename(
                next(s for s in self.sections if s.title == section.parent)
            )
            if parent_file.exists():
                content = self._read_file_with_encoding(parent_file)
                # Extract key points from parent content
                key_points = self._extract_key_points(content)
                return f"\nParent Section Context ({section.parent}):\n{key_points}"
        except Exception as e:
            logger.error(f"Error getting parent context: {e}")
        
        return ""

    def _extract_key_points(self, content: str, max_points: int = 3) -> str:
        """Extract key points from content."""
        # Split into paragraphs and find those starting with important markers
        paragraphs = content.split('\n\n')
        key_points = []
        
        for para in paragraphs:
            if any(marker in para.lower() for marker in 
                ['important', 'key', 'note', 'remember', 'essential']):
                key_points.append(para)
                
        return "\n".join(key_points[:max_points])

    def _create_basic_prompt(self, section: Section, context: str) -> str:
        """Create a basic prompt for content generation.
        
        Args:
            section: Section to generate content for
            context: Full context string
            
        Returns:
            str: Complete prompt with instructions
        """
        return f"""{context}

Write a comprehensive tutorial section about '{section.title}'.

Requirements:
1. Build directly on concepts from previous chapters
2. Maintain consistent terminology and approach
3. Reference specific examples or concepts from earlier chapters where relevant

Content structure:
1. Clear introduction and overview
2. Detailed explanations with examples
3. Common pitfalls and solutions
4. Practice exercises or key takeaways
5. Summary connecting to overall learning path

Format the response in markdown."""

    def _create_specific_prompt(self, section: Section, context: str, is_introduction: bool = False, 
                               part_number: int = None, total_parts: int = None) -> str:
        """Create a content-type specific prompt for introductions.
        
        Args:
            section: Section to generate content for
            context: Context information
            is_introduction: Whether this is an introduction
            part_number: Current part number for multi-part generation
            total_parts: Total number of parts
            
        Returns:
            str: Specialized prompt for introductions
        """
        content_type = self._determine_tutorial_type()
        
        # Create introduction-specific instructions
        intro_instructions = """
INTRODUCTION GUIDELINES:
1. Provide a clear overview of what will be covered in this chapter
2. Explain why this topic is important and relevant
3. Outline the key concepts that will be introduced
4. Set proper expectations for the reader
5. Connect to previous chapters where appropriate (without extensive recaps)
6. Provide a roadmap for the chapter's structure
"""

        # Add part-specific instructions if applicable
        part_instructions = ""
        if part_number is not None and total_parts is not None:
            part_instructions = f"""
IMPORTANT: This is part {part_number} of {total_parts} for this introduction. 
"""

        # Create the prompt
        prompt = f"""You are an expert author writing a comprehensive textbook on {content_type}.
Write the introduction for the chapter titled "{section.title}".{part_instructions}

CONTEXT INFORMATION:
{context}

{intro_instructions}

Write a comprehensive introduction for the chapter titled "{section.title}".
"""
        
        return prompt
