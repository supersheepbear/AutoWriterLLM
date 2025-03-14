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
import os

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
        
        # Initialize book structure
        self.book_structure = {"title": "", "sections": []}
        
        # Create summaries directory
        self.summaries_dir = self.output_dir / "summaries"
        self.summaries_dir.mkdir(exist_ok=True, parents=True)

        # Load configuration
        self.config = self._load_config()
        
        # Set current model from config
        provider_name = self.config.get("provider")
        provider_config = self.config.get("providers", {})
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
                max_context_length=8192,  # Default to model's typical context length
                collection_name=vector_db_config.get("collection_name", "book_content"),
                embedding_model=vector_db_config.get("embedding_model", "all-MiniLM-L6-v2")
            )
            logger.info(f"Initialized context manager with collection: {vector_db_config.get('collection_name', 'book_content')}")
        else:
            self.context_manager = None
            logger.info("Vector database disabled in configuration")
        
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
            
            # Initialize book structure
            self.book_structure = {"title": "", "sections": []}

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

                if level == 1:  # Book title
                    self.book_structure["title"] = title
                    logger.debug(f"Set book title: {title}")
                
                elif level == 2:  # Main section
                    current_section = Section(title=title, subsections=[], level=level)
                    self.sections.append(current_section)
                    
                    # Add to book structure
                    section_data = {
                        "title": title,
                        "level": level,
                        "subsections": []
                    }
                    self.book_structure["sections"].append(section_data)
                    logger.debug(f"Added main section: {title}")

                elif level == 3 and current_section:  # Subsection
                    current_section.subsections.append(title)
                    
                    # Add to book structure
                    if self.book_structure["sections"]:
                        subsection_data = {
                            "title": title,
                            "level": level
                        }
                        self.book_structure["sections"][-1]["subsections"].append(subsection_data)
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

    def generate_section_content(self, section: str, section_level: int, language: str = "auto", 
                             current_unit: int = None, total_units: int = None, progress_callback = None) -> str:
        """生成章节内容。
        
        Args:
            section: 章节标题
            section_level: 章节级别（2表示主章节，3表示子章节）
            language: 内容语言（"auto", "English", "Chinese"）
            current_unit: 当前单元编号
            total_units: 总单元数
            progress_callback: 进度回调函数
            
        Returns:
            str: 生成的内容文件路径
            
        Raises:
            ContentGenerationError: 如果内容生成失败
        """
        # 验证提供者和模型配置
        provider_name = self.config.get("provider")
        if not provider_name:
            raise ContentGenerationError("未配置内容提供者")
        
        # 创建文件名
        filename = self._create_filename(section, section_level)
        output_file = os.path.join(self.output_dir, filename)
        
        # 检查文件是否已存在
        if os.path.exists(output_file):
            logger.info(f"文件 {output_file} 已存在，跳过生成")
            return output_file
        
        # 确定是主章节还是子章节
        is_main_chapter = section_level == 2
        
        # 获取总部分数
        total_parts = 1 if is_main_chapter else self.config.get("parts_per_section", 3)
        
        logger.info(f"开始为 {section} 生成内容，总部分数：{total_parts}")
        
        # 初始化文件
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(f"# {section}\n\n")
        
        try:
            # 获取完整的目录结构
            toc_content = self._get_full_toc_content()
            
            # 存储已生成的内容
            generated_content = []
            
            # 分多次生成内容
            for part in range(1, total_parts + 1):
                # 等待速率限制（如果需要）
                if hasattr(self, 'rate_limiter'):
                    self.rate_limiter.wait_if_needed()
                
                # 获取之前生成的所有内容
                previous_content = "\n\n".join(generated_content) if generated_content else None
                
                # 创建提示
                prompt = self._create_part_prompt(
                    section_title=section,
                    section_level=section_level,
                    part=part,
                    total_parts=total_parts,
                    toc=toc_content,
                    previous_content=previous_content,
                    language=language
                )
                
                # 生成内容
                content = self._generate_with_retry(prompt)
                
                # 移除可能生成的结论部分
                content = self._remove_conclusion_section(content)
                
                # 处理子章节标题
                if is_main_chapter:
                    # 对于主章节，移除可能生成的子章节标题
                    content = self._remove_subsection_headers(content)
                
                # 存储生成的内容
                generated_content.append(content)
                
                # 写入文件
                with open(output_file, 'a', encoding='utf-8') as f:
                    f.write(content)
                    f.write("\n\n")  # 在每个部分之间添加空行
                    f.flush()
                
                # 更新进度
                if progress_callback and current_unit is not None and total_units is not None:
                    progress = (current_unit - 1 + (part / total_parts)) / total_units
                    progress_callback(current_unit, total_units, progress)
            
            logger.info(f"成功生成 {section} 的内容，保存到 {output_file}")
            return output_file
            
        except Exception as e:
            logger.error(f"生成 {section} 内容时出错：{str(e)}")
            if os.path.exists(output_file):
                with open(output_file, 'a', encoding='utf-8') as f:
                    f.write(f"\n\n生成内容时出错：{str(e)}")
            raise ContentGenerationError(f"生成 {section} 内容时出错：{str(e)}")

    def _remove_subsection_headers(self, content: str) -> str:
        """移除内容中的子章节标题。
        
        Args:
            content: 要处理的内容
            
        Returns:
            str: 处理后的内容
        """
        try:
            # 移除二级和三级标题（## 和 ###）
            lines = content.split('\n')
            filtered_lines = []
            skip_next_line = False
            
            for i, line in enumerate(lines):
                # 检查是否是二级或三级标题
                if re.match(r'^#{2,3}\s+', line):
                    skip_next_line = True  # 跳过标题下面的空行
                    continue
                
                if skip_next_line:
                    skip_next_line = False
                    if not line.strip():  # 如果是空行，跳过
                        continue
                
                filtered_lines.append(line)
            
            return '\n'.join(filtered_lines)
            
        except Exception as e:
            logger.error(f"移除子章节标题时出错: {e}")
            return content

    def _remove_conclusion_section(self, content: str) -> str:
        """Remove any conclusion sections and transition phrases from the content.
        
        Args:
            content: The content to process
            
        Returns:
            str: Content with conclusion sections and transition phrases removed
        """
        # Remove sections that start with ## Conclusion or ## Summary
        content = re.sub(r'\n##\s*(Conclusion|Summary|结论|总结).*?(?=\n#|$)', '', content, flags=re.DOTALL|re.IGNORECASE)
        
        # Remove sections that start with # Conclusion or # Summary
        content = re.sub(r'\n#\s*(Conclusion|Summary|结论|总结).*?(?=\n#|$)', '', content, flags=re.DOTALL|re.IGNORECASE)
        
        # Remove any paragraphs containing conclusion-related words
        content = re.sub(r'\n\n.*?(conclusion|summary|总结|结论).*?\n\n', '\n\n', content, flags=re.IGNORECASE)
        
        # Remove transition phrases
        transition_patterns = [
            r'继续阅读下一部分.*?[。\.]',
            r'在下一部分中.*?[。\.]',
            r'在接下来的部分.*?[。\.]',
            r'在下一章中.*?[。\.]',
            r'在下一节中.*?[。\.]',
            r'我们将在下一部分.*?[。\.]',
            r'在前面的部分中.*?[。\.]',
            r'在上一部分中.*?[。\.]',
            r'我们已经学习了.*?[。\.]',
            r'我们已经讨论了.*?[。\.]',
            r'In the next part.*?[。\.]',
            r'In the next section.*?[。\.]',
            r'We will continue.*?[。\.]',
            r'We have discussed.*?[。\.]',
            r'We have learned.*?[。\.]',
            r'In the previous part.*?[。\.]',
            r'In the previous section.*?[。\.]'
        ]
        
        for pattern in transition_patterns:
            content = re.sub(pattern, '', content, flags=re.IGNORECASE)
        
        return content.strip() + "\n"

    def _create_part_prompt(self, section_title: str, section_level: int, part: int, total_parts: int, 
                          toc: str, previous_content: str = None, language: str = "auto") -> str:
        """创建特定部分的提示。
        
        Args:
            section_title: 章节标题
            section_level: 章节级别（2表示主章节，3表示子章节）
            part: 当前部分编号
            total_parts: 总部分数
            toc: 目录内容
            previous_content: 之前生成的内容
            language: 内容语言
            
        Returns:
            str: 生成提示
        """
        # 获取章节编号
        chapter_match = re.match(r'^(\d+)\.(\d+)?', section_title)
        if not chapter_match:
            raise ValueError(f"Invalid section title format: {section_title}")
        
        main_chapter = chapter_match.group(1)
        sub_chapter = chapter_match.group(2) if chapter_match.group(2) else None
        
        # 从目录中提取书籍主题
        book_title = ""
        for line in toc.split('\n'):
            if line.startswith('# '):
                book_title = line.lstrip('# ').strip()
                break
                
        # 读取书籍摘要
        summary_file = self.output_dir / "book_summary.md"
        book_summary = ""
        if summary_file.exists():
            try:
                with open(summary_file, 'r', encoding='utf-8') as f:
                    book_summary = f.read().strip()
            except Exception as e:
                logger.error(f"读取书籍摘要时出错: {e}")
        
        if section_level == 2:  # 主章节
            part_instruction = f"""这是关于"{book_title}"的教材第{main_chapter}章的主要内容。

【书籍摘要】
{book_summary}

【完整目录】
{toc}

【重要提示】
本次请求仅生成第{main_chapter}章的内容。不要生成其他章节的内容。

【章节结构要求】
1. 主章节应提供高层次概述
2. 重点介绍将在子章节中详细讨论的关键概念
3. 不包含详细示例或练习
4. 不创建子章节标题
5. 严格遵守章节编号，只生成第{main_chapter}章的内容

【内容要求】
1. 确保内容严格围绕"{book_title}"这个主题
2. 清晰介绍本章主题及其在整体主题中的位置
3. 解释概念的重要性和相关性
4. 概述子章节将涵盖的内容
5. 保持内容简洁，专注于主要思想
6. 不要生成其他章节的内容
7. 不要使用过渡短语引用其他章节
"""
        else:  # 子章节
            current_section = f"{main_chapter}.{sub_chapter}"
            
            # 根据part编号设置不同的主题焦点
            focus_topics = {
                1: "基础概念和定义",
                2: "深入分析和应用",
                3: "高级特性和实践"
            }
            
            current_focus = focus_topics.get(part, "补充内容")
            
            part_instruction = f"""这是"{book_title}"教材中子章节 {section_title} 的第 {part} 部分（共 {total_parts} 部分）。

【完整目录】
{toc}

【重要提示】
本次请求仅生成章节编号 {current_section} 的内容。不要生成其他编号的内容。

【书籍摘要】
{book_summary}

【本节重点】：{current_focus}

【章节编号】：{current_section}

【结构要求】
1. 仅使用三级标题（{current_section}）
2. 不要创建更深层次的标题
3. 不要使用过渡短语
4. 保持内容的独立性和完整性
5. 严格遵守章节编号，只生成 {current_section} 的内容
6. 不要生成其他章节的内容
7. 不要使用过渡短语引用其他章节
"""

            # 添加之前生成的内容作为上下文
            if previous_content:
                part_instruction += f"\n\n【已生成的内容】\n{previous_content}\n\n请确保新内容与已有内容自然衔接，但不要直接引用或重复。"

            # 只在最后一部分添加练习题要求
            if part == total_parts:
                part_instruction += """
【练习题要求】
1. 在本节末尾添加练习题
2. 包含理论和实践题目
3. 提供清晰的题目说明
4. 难度适中但具有挑战性
5. 不要在正文中包含答案
"""
            else:
                part_instruction += "\n注意：本部分不包含练习题。"
        
        # 添加语言指示
        if language.lower() == "chinese":
            part_instruction += "\n\n请用中文编写内容。"
        elif language.lower() == "english":
            part_instruction += "\n\nPlease write the content in English."
            
        return part_instruction

    def _get_main_chapter_content(self, main_chapter_title: str) -> str:
        """获取主章节内容。
        
        Args:
            main_chapter_title: 主章节标题
            
        Returns:
            str: 主章节内容，如果找不到则返回空字符串
        """
        try:
            # 创建主章节文件名
            filename = self._create_filename(main_chapter_title, 2)
            file_path = os.path.join(self.output_dir, filename)
            
            # 检查文件是否存在
            if not os.path.exists(file_path):
                logger.warning(f"找不到主章节文件：{file_path}")
                return ""
            
            # 读取文件内容
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 移除标题（如果存在）
            content_lines = content.split('\n')
            if content_lines and content_lines[0].startswith('# '):
                content = '\n'.join(content_lines[1:])
            
            return content.strip()
            
        except Exception as e:
            logger.error(f"获取主章节内容时出错: {e}")
            return ""

    def _get_previous_section_content(self, current_section: Section) -> str:
        """获取上一小节的内容。
        
        Args:
            current_section: 当前章节
            
        Returns:
            str: 上一小节内容
        """
        try:
            # 从章节标题中提取编号
            section_match = re.match(r'^(\d+\.\d+)(\.\d+)?', current_section.title)
            if not section_match:
                return "无法解析章节编号"
                
            section_base = section_match.group(1)  # 例如 "1.1"
            section_sub = section_match.group(2)   # 例如 ".1" 或 None
            
            # 如果是三级标题 (例如 1.1.1)
            if section_sub:
                current_num = int(section_sub.replace(".", ""))
                if current_num <= 1:
                    return "这是本小节的第一个部分，没有前序内容"
                    
                # 构建上一个三级标题
                prev_title = f"{section_base}.{current_num - 1}"
                
                # 查找对应的章节文件
                for section in self.sections:
                    if section.level == 2:  # 主章节
                        for subsection in section.subsections:
                            if subsection.startswith(prev_title):
                                prev_section = Section(title=subsection, subsections=[], level=3, parent=section.title)
                                prev_file = self.output_dir / self._create_filename(prev_section)
                                if prev_file.exists():
                                    content = self._read_file_with_encoding(prev_file)
                                    summary = content[:1000] + ("..." if len(content) > 1000 else "")
                                    return f"上一小节 {prev_title} 内容：\n{summary}\n\n请确保您生成的内容与上一小节的内容自然衔接，但不要直接引用或回顾上一小节。"
            
            # 如果是二级标题 (例如 1.1)
            else:
                base_parts = section_base.split(".")
                if len(base_parts) >= 2:
                    chapter = int(base_parts[0])
                    section = int(base_parts[1])
                    
                    if section <= 1:
                        return "这是本章的第一个小节，没有前序内容"
                        
                    # 构建上一个二级标题
                    prev_title = f"{chapter}.{section - 1}"
                    
                    # 查找对应的章节文件
                    for main_section in self.sections:
                        if main_section.level == 2:  # 主章节
                            for subsection in main_section.subsections:
                                if subsection.startswith(prev_title):
                                    prev_section = Section(title=subsection, subsections=[], level=3, parent=main_section.title)
                                    prev_file = self.output_dir / self._create_filename(prev_section)
                                    if prev_file.exists():
                                        content = self._read_file_with_encoding(prev_file)
                                        summary = content[:1000] + ("..." if len(content) > 1000 else "")
                                        return f"上一小节 {prev_title} 内容：\n{summary}\n\n请确保您生成的内容与上一小节的内容自然衔接，但不要直接引用或回顾上一小节。"
            
            return "无法找到上一小节内容"
            
        except Exception as e:
            logger.error(f"获取上一小节内容时出错: {e}")
            return "获取上一小节内容时出错"

    def _get_full_toc_content(self) -> str:
        """获取完整的目录结构。
        
        Returns:
            str: 格式化的目录结构
        """
        try:
            # 获取所有章节
            sections = self.book_structure.get("sections", [])
            
            # 构建目录
            toc_lines = []
            
            # 添加书名
            book_title = self.book_structure.get("title", "")
            if book_title:
                toc_lines.append(f"# {book_title}")
                toc_lines.append("")
            
            # 添加章节
            for section in sections:
                title = section.get("title", "")
                level = section.get("level", 2)
                
                if level == 2:  # 主章节
                    toc_lines.append(f"## {title}")
                    
                    # 添加子章节
                    subsections = section.get("subsections", [])
                    for subsection in subsections:
                        sub_title = subsection.get("title", "")
                        toc_lines.append(f"### {sub_title}")
            
            return "\n".join(toc_lines)
            
        except Exception as e:
            logger.error(f"获取目录结构时出错: {e}")
            return "无法获取目录结构"
            
    def _get_recent_chapters_content(self, current_section: Section) -> str:
        """获取最近章节的内容摘要。
        
        Args:
            current_section: 当前正在生成的章节
            
        Returns:
            str: 最近章节的内容摘要
        """
        try:
            # 找到当前章节的索引
            current_index = next(
                (i for i, s in enumerate(self.sections) if s.title == current_section.title),
                -1
            )
            
            if current_index <= 0:
                return "这是第一个章节，没有前序内容"
                
            # 获取最近的2个章节
            recent_sections = self.sections[max(0, current_index-2):current_index]
            
            content_parts = []
            for section in recent_sections:
                file_path = self.output_dir / self._create_filename(section)
                if file_path.exists():
                    content = self._read_file_with_encoding(file_path)
                    # 提取前1000个字符作为摘要
                    summary = content[:1000] + ("..." if len(content) > 1000 else "")
                    content_parts.append(f"章节 {section.title} 内容摘要:\n{summary}")
            
            if not content_parts:
                return "没有找到前序章节内容"
                
            return "\n\n".join(content_parts)
        except Exception as e:
            logger.error(f"获取最近章节内容时出错: {e}")
            return "获取最近章节内容时出错"

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
- DO NOT add transition phrases like "In the next part, we will..."
- DO NOT end with sentences that lead to the next part
"""
            elif part_number < total_parts:
                part_instructions += f"""
For part {part_number}:
- Continue directly from where part {part_number-1} left off
- DO NOT recap what was covered in previous parts
- DO NOT use phrases like "As we saw in the previous part"
- Focus on new content and examples
- Explore concepts in greater depth
- DO NOT include a summary at the end of this part
- DO NOT add transition phrases to the next part
"""
            else:  # Last part
                part_instructions += f"""
For this final part:
- Continue directly from where part {part_number-1} left off
- DO NOT recap what was covered in previous parts
- Complete any remaining explanations and examples
- DO NOT include a conclusion or summary section
- DO NOT use phrases like "In this chapter, we have learned"
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
9. DO NOT add transition phrases between parts
10. DO NOT include a conclusion or summary section
11. DO NOT end with sentences that lead to the next section
12. Focus on providing comprehensive, in-depth explanations
"""

        # Add content-specific guidelines based on the type
        if content_type == "programming":
            prompt += """
13. Include well-commented code examples
14. Explain algorithms and data structures clearly
15. Cover best practices and common pitfalls
16. Include debugging tips and performance considerations
17. Provide multiple code examples for each concept
"""
        elif content_type == "science":
            prompt += """
13. Include scientific principles and theories
14. Explain experimental methods and results
15. Reference important studies and findings
16. Include diagrams of scientific processes
17. Provide multiple examples of scientific applications
"""
        elif content_type == "mathematics":
            prompt += """
13. Include clear mathematical notation using LaTeX
14. Provide step-by-step derivations of formulas
15. Include visual representations of mathematical concepts
16. Provide practice problems with solutions
17. Offer multiple approaches to solving problems
"""
        elif content_type == "history":
            prompt += """
13. Include accurate dates and chronology
14. Provide context for historical events
15. Include perspectives from different historical sources
16. Connect historical events to broader themes
17. Analyze historical events in depth
"""
        elif content_type == "literature":
            prompt += """
13. Include analysis of literary techniques
14. Provide examples from important texts
15. Discuss cultural and historical context of works
16. Include biographical information about key authors
17. Offer multiple interpretations of literary works
"""
        elif content_type == "business":
            prompt += """
13. Include case studies of real businesses
14. Provide practical frameworks and models
15. Include market analysis and trends
16. Discuss ethical considerations in business
17. Offer multiple business strategy examples
"""
        elif content_type == "arts":
            prompt += """
13. Include descriptions of artistic techniques
14. Provide historical context for artistic movements
15. Include analysis of notable works
16. Discuss the cultural impact of art
17. Offer multiple examples of artistic styles
"""
        elif content_type == "medicine":
            prompt += """
13. Include accurate medical terminology
14. Describe diagnostic criteria and treatment protocols
15. Include anatomical diagrams and illustrations
16. Discuss current research and evidence-based practices
17. Provide multiple clinical examples
"""
        elif content_type == "philosophy":
            prompt += """
13. Include clear explanations of philosophical concepts
14. Provide historical context for philosophical ideas
15. Present different perspectives on philosophical questions
16. Include thought experiments and logical arguments
17. Analyze philosophical arguments in depth
"""
        elif content_type == "language":
            prompt += """
13. Include examples of language usage
14. Provide clear explanations of grammatical rules
15. Include pronunciation guides where relevant
16. Discuss cultural aspects of language
17. Offer multiple examples of language patterns
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

    def _create_filename(self, section_title: str, section_level: int = 2) -> str:
        """创建章节内容的文件名。
        
        Args:
            section_title: 章节标题
            section_level: 章节级别（2表示主章节，3表示子章节）
            
        Returns:
            str: 文件名
        """
        try:
            # 提取章节编号
            if section_level == 2:  # 主章节
                # 尝试从各种格式中提取编号（包括中文）
                chapter_match = re.search(r'^(\d+)[\.。\s]', section_title)
                if chapter_match:
                    chapter_num = chapter_match.group(1)
                return f"chapter-{chapter_num}.md"
                
                # 尝试匹配中文数字（一、二、三等）
                chinese_num_match = re.search(r'^[第]?([一二三四五六七八九十]+)[章節节篇]', section_title)
                if chinese_num_match:
                    chinese_num = chinese_num_match.group(1)
                    # 将中文数字转换为阿拉伯数字
                    chinese_to_arabic = {
                        '一': '1', '二': '2', '三': '3', '四': '4', '五': '5',
                        '六': '6', '七': '7', '八': '8', '九': '9', '十': '10'
                    }
                    if chinese_num in chinese_to_arabic:
                        return f"chapter-{chinese_to_arabic[chinese_num]}.md"
            else:  # 子章节
                # 尝试匹配格式如 "1.1" 或 "第1.1节" - 改进正则以更好地匹配
                subchapter_match = re.search(r'(\d+)\.(\d+)[\.。\s]', section_title)
                if not subchapter_match:
                    # 尝试其他可能的格式
                    subchapter_match = re.search(r'(\d+)\.(\d+)', section_title)
                
                if subchapter_match:
                    chapter_num = subchapter_match.group(1)
                    section_num = subchapter_match.group(2)
                    return f"chapter-{chapter_num}-{section_num}.md"
                
                # 尝试匹配中文格式
                chinese_subchapter_match = re.search(r'([一二三四五六七八九十]+)\.([一二三四五六七八九十]+)', section_title)
                if chinese_subchapter_match:
                    chinese_to_arabic = {
                        '一': '1', '二': '2', '三': '3', '四': '4', '五': '5',
                        '六': '6', '七': '7', '八': '8', '九': '9', '十': '10'
                    }
                    ch_num = chinese_subchapter_match.group(1)
                    sec_num = chinese_subchapter_match.group(2)
                    if ch_num in chinese_to_arabic and sec_num in chinese_to_arabic:
                        return f"chapter-{chinese_to_arabic[ch_num]}-{chinese_to_arabic[sec_num]}.md"
            
            # 如果无法提取编号，记录详细信息并使用标题的哈希值
            logger.warning(f"无法从标题 '{section_title}' 提取章节编号（级别：{section_level}），使用哈希值")
            title_hash = hashlib.md5(section_title.encode()).hexdigest()[:8]
            return f"chapter-{title_hash}.md"
                
        except Exception as e:
            logger.error(f"创建文件名时出错: {e}")
            # 使用标题的哈希值作为备用
            title_hash = hashlib.md5(section_title.encode()).hexdigest()[:8]
            return f"chapter-{title_hash}.md"

    def _convert_chinese_numerals(self, text: str) -> str:
        """Convert Chinese numerals to Arabic numerals.
        
        Args:
            text: Text containing Chinese numerals
            
        Returns:
            str: Text with Chinese numerals converted to Arabic numerals
        """
        chinese_numerals = {
            '零': '0', '一': '1', '二': '2', '三': '3', '四': '4',
            '五': '5', '六': '6', '七': '7', '八': '8', '九': '9',
            '十': '10', '百': '100', '千': '1000', '万': '10000'
        }
        
        # If no Chinese characters, return as is
        if not re.search(r'[\u4e00-\u9fff]', text):
            return text
            
        result = text
        for cn, ar in chinese_numerals.items():
            result = result.replace(cn, ar)
            
        # Try to evaluate simple expressions (e.g., "一百二十" -> "120")
        try:
            # Extract numbers
            numbers = re.findall(r'\d+', result)
            if numbers:
                # If we have multiple numbers, try to combine them
                value = 0
                for num in numbers:
                    value = value * 100 + int(num)
                return str(value)
            return result
        except Exception as e:
            logger.warning(f"Failed to convert complex Chinese numeral: {text}")
            return text

    def generate_all_content(self, progress_callback=None, language="auto"):
        """生成所有章节的内容。
        
        Args:
            progress_callback: 可选的进度回调函数
            language: 内容语言（"auto", "English", "Chinese"）
            
        Returns:
            bool: 是否成功生成所有内容
            
        Raises:
            ContentGenerationError: 如果内容生成失败
        """
        # 获取所有章节
        sections = self.book_structure.get("sections", [])
        
        # 计算总单元数（主章节和子章节）
        main_chapters = []
        sub_chapters = []
        
        for section in sections:
            # 提取章节标题和级别
            title = section.get("title", "")
            level = section.get("level", 2)
            
            if level == 2:  # 主章节
                main_chapters.append(title)
                
                # 获取子章节
                subsections = section.get("subsections", [])
                for subsection in subsections:
                    sub_title = subsection.get("title", "")
                    sub_chapters.append(sub_title)
        
        total_units = len(main_chapters) + len(sub_chapters)
        current_unit = 0
        
        logger.info(f"开始生成内容，共 {len(main_chapters)} 个主章节和 {len(sub_chapters)} 个子章节，语言：{language}")
        
        # 首先生成所有主章节
        for title in main_chapters:
            current_unit += 1
            
            try:
                # 检查文件是否已存在
                filename = self._create_filename(title, 2)
                output_file = os.path.join(self.output_dir, filename)
                
                if os.path.exists(output_file):
                    logger.info(f"主章节 {title} 的文件已存在，跳过生成")
                    if progress_callback:
                        progress_callback(current_unit, total_units)
                    continue
                
                # 生成主章节内容
                logger.info(f"生成主章节 {title} 的内容")
                self.generate_section_content(
                    section=title,
                    section_level=2,
                    language=language,
                    current_unit=current_unit,
                    total_units=total_units,
                    progress_callback=progress_callback
                )
                
                if progress_callback:
                    progress_callback(current_unit, total_units)
                    
            except Exception as e:
                logger.error(f"生成主章节 {title} 时出错：{str(e)}")
                raise ContentGenerationError(f"生成主章节 {title} 失败：{str(e)}")
        
        # 然后生成所有子章节
        for title in sub_chapters:
            current_unit += 1
            
            try:
                # 检查文件是否已存在
                filename = self._create_filename(title, 3)
                output_file = os.path.join(self.output_dir, filename)
                
                if os.path.exists(output_file):
                    logger.info(f"子章节 {title} 的文件已存在，跳过生成")
                    if progress_callback:
                        progress_callback(current_unit, total_units)
                    continue
                
                # 生成子章节内容
                logger.info(f"生成子章节 {title} 的内容")
                self.generate_section_content(
                    section=title,
                    section_level=3,
                    language=language,
                    current_unit=current_unit,
                    total_units=total_units,
                    progress_callback=progress_callback
                )
                
                if progress_callback:
                    progress_callback(current_unit, total_units)
                    
            except Exception as e:
                logger.error(f"生成子章节 {title} 时出错：{str(e)}")
                raise ContentGenerationError(f"生成子章节 {title} 失败：{str(e)}")
        
        logger.info(f"成功生成所有内容，共 {total_units} 个单元")
        return True

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

    def _create_book_planning_prompt(self, user_input: str, level: str = "intermediate", language: str = "auto") -> str:
        """Create a prompt for generating a book plan.
        
        Args:
            user_input: User's description of the book they want to create
            level: The target level (beginner, intermediate, advanced)
            language: Language to generate content in ("auto", "English", "Chinese")
            
        Returns:
            str: The prompt for the AI
        """
        # 直接使用用户输入作为主题
        topic = user_input
        
        # Language-specific instructions
        language_instruction = ""
        if language == "English":
            language_instruction = "Generate the response in English, regardless of the input language."
        elif language == "Chinese":
            language_instruction = "Generate the response in Chinese (中文), regardless of the input language."
        else:  # auto
            language_instruction = "Support both English and Chinese content. If the topic is in Chinese, generate the response in Chinese."
        
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
7. {language_instruction}

FORMAT YOUR RESPONSE EXACTLY AS FOLLOWS:

```markdown:table_of_contents.md
# Book Title (书名)
## 1. Chapter Title (第一章)
### 1.1 Section Title (第一节)
### 1.2 Section Title (第二节)
## 2. Chapter Title (第二章)
...
```

```markdown:book_summary.md
[Write a 3-5 paragraph summary of the book here, including purpose, audience, and prerequisites. {language_instruction}]
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
            
        Note:
            Handles both English and Chinese content in markdown blocks.
            Supports various markdown code block formats.
        """
        try:
            # Look for markdown code blocks with or without file labels
            patterns = [
                # Try with file labels first
                (r"```markdown:table_of_contents\.md\n(.*?)```", r"```markdown:book_summary\.md\n(.*?)```"),
                # Fallback to simple markdown blocks
                (r"```markdown\n# table_of_contents\.md\n(.*?)```", r"```markdown\n# book_summary\.md\n(.*?)```"),
                # Another common format
                (r"```\n# table_of_contents\.md\n(.*?)```", r"```\n# book_summary\.md\n(.*?)```"),
                # Most basic format
                (r"```markdown\n(.*?)```.*?```markdown\n(.*?)```", re.DOTALL)
            ]
            
            for toc_pattern, summary_pattern in patterns:
                toc_match = re.search(toc_pattern, content, re.DOTALL)
                summary_match = re.search(summary_pattern, content, re.DOTALL)
                
                if toc_match and summary_match:
                    toc_content = toc_match.group(1).strip()
                    summary_content = summary_match.group(1).strip()
                    
                    # Validate the content structure
                    if not self._validate_toc_structure(toc_content):
                        logger.warning("Invalid TOC structure detected")
                        continue
                        
                    logger.debug("Successfully parsed AI response using pattern")
                    return toc_content, summary_content
            
            # If no patterns matched, try to find any markdown blocks
            markdown_blocks = re.findall(r"```(?:markdown)?\n(.*?)```", content, re.DOTALL)
            if len(markdown_blocks) >= 2:
                logger.warning("Using fallback pattern matching for markdown blocks")
                toc_content = markdown_blocks[0].strip()
                summary_content = markdown_blocks[1].strip()
                
                # Validate the content structure
                if self._validate_toc_structure(toc_content):
                    return toc_content, summary_content
            
            raise ValueError("无法在AI响应中找到正确格式的目录或摘要")
            
        except Exception as e:
            logger.error(f"Error parsing AI response: {e}")
            logger.debug(f"Raw content: {content[:500]}...")  # Log first 500 chars of content
            raise ValueError(f"解析AI响应失败: {e}")
            
    def _validate_toc_structure(self, content: str) -> bool:
        """Validate the structure of the table of contents.
        
        Args:
            content: TOC content to validate
            
        Returns:
            bool: True if valid, False otherwise
        """
        try:
            lines = content.strip().split('\n')
            if not lines:
                return False
                
            # Check if first line is a title (starts with single #)
            if not re.match(r'^#\s+[\u4e00-\u9fff\w]', lines[0]):
                return False
                
            # Check if there are chapter headings (starts with ##)
            has_chapters = False
            for line in lines[1:]:
                if re.match(r'^##\s+[\u4e00-\u9fff\w]', line):
                    has_chapters = True
                    break
                    
            return has_chapters
            
        except Exception as e:
            logger.error(f"Error validating TOC structure: {e}")
            return False

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
        """使用自动重试机制生成内容。
        
        Args:
            prompt: 发送给AI的提示
            max_retries: 失败时的最大重试次数
            
        Returns:
            str: 生成的内容
            
        Raises:
            ContentGenerationError: 如果所有重试后内容生成仍然失败
        """
        # 直接使用客户端而不是创建新的APIContentGenerator
        provider_name = self.config.get("provider")
        providers_config = self.config.get("providers", {})
        provider_config = providers_config.get(provider_name, {})
        
        # 获取当前模型配置
        current_model = None
        model_name = self.current_model
        
        # 在提供者的模型列表中查找当前模型
        for model in provider_config.get("models", []):
            if model.get("name") == model_name:
                current_model = model
                break
        
        # 如果找不到模型，使用列表中的第一个模型
        if not current_model and provider_config.get("models"):
            current_model = provider_config["models"][0]
            model_name = current_model.get("name")
            logger.warning(f"找不到模型 {self.current_model}，使用 {model_name}")
        
        # 获取模型参数
        max_tokens = current_model.get("max_tokens") if current_model else self.config.get("max_tokens_per_part", 8192)
        temperature = current_model.get("temperature") if current_model else 0.7
        
        # 系统消息，指导AI关于内容生成的规则
        system_message = """您是一位专业教科书作者。在生成内容时，请遵循以下规则：

【章节结构规则】
1. 主章节（如1.0, 2.0）只包含概述，不包含详细内容或子章节
2. 子章节（如1.1, 1.2）包含详细内容，但不创建更深层次的标题
3. 三级章节（如1.1.1, 1.1.2）是最小内容单元，包含详细讲解
4. 练习题只在最后一个小节中出现
5. 每个请求只生成一个特定章节的内容

【章节编号规则】
1. 主章节使用单个数字编号（如1, 2, 3）
2. 子章节使用两级编号（如1.1, 1.2, 2.1）
3. 三级章节使用三级编号（如1.1.1, 1.1.2, 2.1.1）
4. 不要创建四级或更深层次的编号（如1.1.1.1）

【格式规则】
1. 不要包含结论或总结部分
2. 不要添加部分之间或章节之间的过渡短语
3. 不要以引导到下一部分或章节的句子结尾
4. 不要使用"在下一部分中"或"正如我们将在下一章中看到的"等表达
5. 不要创建四级或更深层次的标题（如1.1.1.1）
6. 每个章节内容应该是完整的，不依赖于其他章节

【参考规则】
1. 参考完整的目录结构来理解内容的整体组织
2. 参考主章节内容来确保子章节内容与主题一致
3. 参考上一小节内容来确保内容的连贯性
4. 不要直接引用或回顾上一小节内容

请严格按照提示中的要求和格式生成内容。"""
        
        # 应用速率限制（如果可用）
        if hasattr(self, 'rate_limiter'):
            self.rate_limiter.wait_if_needed()
        
        for attempt in range(max_retries):
            try:
                logger.info(f"生成尝试 {attempt + 1}/{max_retries}")
                
                # 直接使用客户端
                if provider_name == "anthropic":
                    logger.debug(f"使用Anthropic API，模型：{model_name}")
                    response = self.client.messages.create(
                        model=model_name,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        system=system_message,
                        messages=[{"role": "user", "content": prompt}]
                    )
                    content = response.content[0].text
                    logger.info("成功从Anthropic API生成内容")
                    return content
                    
                elif provider_name == "openai":
                    logger.debug(f"使用OpenAI API，模型：{model_name}")
                    response = self.client.chat.completions.create(
                        model=model_name,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        messages=[
                            {"role": "system", "content": system_message},
                            {"role": "user", "content": prompt}
                        ]
                    )
                    content = response.choices[0].message.content
                    logger.info("成功从OpenAI API生成内容")
                    return content
                    
                else:
                    # 其他提供者的通用API调用
                    logger.debug(f"使用{provider_name} API，模型：{model_name}")
                    
                    # 尝试使用聊天完成端点（OpenAI兼容）
                    try:
                        response = self.client.chat.completions.create(
                            model=model_name,
                            max_tokens=max_tokens,
                            temperature=temperature,
                            messages=[
                                {"role": "system", "content": system_message},
                                {"role": "user", "content": prompt}
                            ]
                        )
                        content = response.choices[0].message.content
                        logger.info(f"成功从{provider_name} API生成内容")
                        return content
                    except (AttributeError, NotImplementedError) as e:
                        # 如果聊天完成不可用，尝试直接HTTP请求
                        logger.debug(f"{provider_name}不支持聊天完成，使用HTTP请求：{e}")
                        
                        url = f"{provider_config.get('base_url')}/chat/completions"
                    headers = {
                            "Authorization": f"Bearer {provider_config.get('api_key')}",
                        "Content-Type": "application/json"
                    }
                    
                    data = {
                            "model": model_name,
                        "messages": [
                                {"role": "system", "content": system_message},
                            {"role": "user", "content": prompt}
                        ],
                            "temperature": temperature,
                            "max_tokens": max_tokens
                    }
                    
                    logger.debug(f"请求数据：{data}")
                    response = requests.post(url, headers=headers, json=data, timeout=120)
                    
                    if response.status_code != 200:
                        logger.error(f"API错误：{response.status_code} - {response.text}")
                        raise ContentGenerationError(f"API错误：{response.status_code}")
                    
                    result = response.json()
                    logger.debug(f"API响应：{result}")
                    
                    if not result.get("choices"):
                        logger.error("API响应中没有选择")
                        raise ContentGenerationError("API响应中没有内容")
                    
                    content = result["choices"][0]["message"]["content"]
                    if not content:
                        logger.error("API响应中内容为空")
                        raise ContentGenerationError("API响应中内容为空")
                    
                    logger.info(f"成功从{provider_name} API生成内容")
                    return content
                    
            except (requests.Timeout, requests.ConnectionError) as e:
                logger.warning(f"尝试{attempt + 1}时网络错误：{e}")
                if attempt == max_retries - 1:
                    raise ContentGenerationError(f"{max_retries}次尝试后网络错误")
                wait_time = min(300, 10 * (2 ** attempt))
                logger.info(f"等待{wait_time}秒后重试...")
                time.sleep(wait_time)
                
            except Exception as e:
                logger.error(f"尝试{attempt + 1}时错误：{e}")
                if attempt == max_retries - 1:
                    raise ContentGenerationError(f"{max_retries}次尝试后失败：{str(e)}")
                wait_time = min(300, 10 * (2 ** attempt))
                logger.info(f"等待{wait_time}秒后重试...")
                time.sleep(wait_time)
        
        raise ContentGenerationError("所有重试后内容生成失败")

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
            - Contains actual text characters (including Chinese, Japanese, Korean)
            - Not just special characters or whitespace
        """
        # Check if empty or too short
        if not topic or len(topic.strip()) < 3:
            logger.warning("Topic validation failed: Too short")
            return False
        
        # Check for meaningful content (including CJK characters)
        # \u4e00-\u9fff: Chinese characters
        # \u3040-\u309F: Hiragana
        # \u30A0-\u30FF: Katakana
        # \uAC00-\uD7AF: Korean Hangul
        # \w: Latin letters, numbers, and underscore
        if not re.search(r'[\u4e00-\u9fff\u3040-\u309F\u30A0-\u30FF\uAC00-\uD7AF\w]', topic):
            logger.warning(f"Topic validation failed: No valid text characters. Topic: {topic}")
            return False
        
        # Check maximum length
        if len(topic) > 500:
            logger.warning("Topic validation failed: Too long")
            return False
        
        logger.debug(f"Topic validation passed: {topic}")
        return True

    def _validate_provider_and_model(self) -> bool:
        """Validate that provider and model are properly configured.
        
        Returns:
            bool: True if valid configuration exists, False otherwise
        """
        if not self.config.get("provider"):
            messagebox.showerror(
                "Configuration Error",
                "No AI provider configured. Please set a provider in config.yaml"
            )
            logger.error("No provider configured")
            return False
            
        provider_name = self.config["provider"]
        provider_config = self.config.get("providers", {}).get(provider_name)
        
        if not provider_config:
            messagebox.showerror(
                "Configuration Error",
                f"Provider '{provider_name}' not found in configuration"
            )
            logger.error(f"Provider {provider_name} not found in config")
            return False
            
        if not self.current_model:
            models = provider_config.get("models", [])
            if not models:
                messagebox.showerror(
                    "Configuration Error",
                    f"No models configured for provider '{provider_name}'"
                )
                logger.error(f"No models found for provider {provider_name}")
                return False
            self.current_model = models[0].get("name")
            
        return True

    def generate_book_plan(self, topic: str, level: str = "intermediate", 
                      provider_name: Optional[str] = None, 
                      model_name: Optional[str] = None,
                      language: str = "auto") -> Tuple[str, str]:
        """Generate a book plan based on the topic.
        
        Args:
            topic: The topic to generate a book plan for
            level: The target level (beginner, intermediate, advanced)
            provider_name: Optional provider name to use (e.g., "anthropic", "openai", "lingyiwanwu")
            model_name: Optional model name to use
            language: Language to generate content in ("auto", "English", "Chinese")
            
        Returns:
            Tuple[str, str]: (table_of_contents, book_summary)
            
        Raises:
            ContentGenerationError: If generation fails
        """
        # If provider_name is provided, temporarily switch to it
        original_provider = None
        original_model = None
        
        try:
            if provider_name:
                original_provider = self.config.get("provider")
                original_model = self.current_model
                self.set_model(provider_name, model_name or self.current_model)
            
            if not self._validate_provider_and_model():
                raise ContentGenerationError("Invalid provider or model configuration")
                
            prompt = self._create_book_planning_prompt(topic, level, language)
            content = self._generate_with_retry(prompt)
            return self._parse_ai_response(content)
            
        except Exception as e:
            logger.error(f"Failed to generate book plan: {e}")
            raise ContentGenerationError(f"Failed to generate book plan: {str(e)}")
            
        finally:
            # Restore original provider and model if they were changed
            if original_provider:
                try:
                    self.set_model(original_provider, original_model)
                except Exception as e:
                    logger.warning(f"Failed to restore original provider and model: {e}")

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

    def _add_learning_metadata(self, content: str,
                         objectives: List[LearningObjective],
                         prerequisites: List[str]) -> str:
        """Add learning objectives and prerequisites to content.
        
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
7. DO NOT include a conclusion or summary section
8. DO NOT add transition phrases like "In the next section, we will..."
9. DO NOT end with sentences that lead to the next section
10. DO NOT use phrases like "As we will see later in this chapter"
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
DO NOT include a conclusion or summary section.
DO NOT add transition phrases between parts or to other sections.
DO NOT end with sentences that lead to the next section or chapter.
"""
        
        return prompt

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

    def _get_parent_chapter_title(self, section_title: str) -> str:
        """从子章节标题中获取父章节标题。
        
        Args:
            section_title: 子章节标题
            
        Returns:
            str: 父章节标题，如果无法确定则返回空字符串
        """
        try:
            # 尝试从格式如 "1.1 子章节标题" 中提取父章节编号
            match = re.search(r'^(\d+)\.', section_title)
            if match:
                chapter_num = match.group(1)
                
                # 查找目录中对应的主章节
                toc_content = self._get_full_toc_content()
                for line in toc_content.split('\n'):
                    # 查找匹配的主章节标题
                    main_chapter_match = re.search(rf'^#{1,2}\s*{chapter_num}[\.:\s]', line)
                    if main_chapter_match:
                        # 提取主章节标题
                        title_match = re.search(r'^#{1,2}\s*(.*?)$', line)
                        if title_match:
                            return title_match.group(1).strip()
            
            # 尝试匹配中文格式
            chinese_match = re.search(r'^第?([一二三四五六七八九十]+)[章節节]', section_title)
            if chinese_match:
                # 在目录中查找对应的主章节
                toc_content = self._get_full_toc_content()
                chinese_num = chinese_match.group(1)
                for line in toc_content.split('\n'):
                    if f'第{chinese_num}章' in line or f'第{chinese_num}節' in line:
                        title_match = re.search(r'^#{1,2}\s*(.*?)$', line)
                        if title_match:
                            return title_match.group(1).strip()
            
            logger.warning(f"无法从标题 '{section_title}' 确定父章节")
            return ""
            
        except Exception as e:
            logger.error(f"获取父章节标题时出错: {e}")
            return ""
