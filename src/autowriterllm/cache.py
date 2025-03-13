"""Cache module for AutoWriterLLM.

This module provides caching functionality to avoid redundant API calls.
"""

import logging
import hashlib
from pathlib import Path
from typing import Optional

from autowriterllm.models import Section

logger = logging.getLogger(__name__)


class ContentCache:
    """Cache for generated content to avoid redundant API calls.
    
    This class provides methods for caching and retrieving generated content,
    which helps reduce API costs and speeds up content generation.
    
    Attributes:
        cache_dir (Path): Directory to store cache files
    """
    
    def __init__(self, cache_dir: Path):
        """Initialize the cache.
        
        Args:
            cache_dir: Directory to store cache files
            
        Raises:
            Exception: If the cache directory cannot be created
        """
        self.cache_dir = Path(cache_dir)
        try:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Initialized content cache at {self.cache_dir}")
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
        key = hashlib.md5(content.encode()).hexdigest()
        logger.debug(f"Generated cache key {key} for section '{section.title}'")
        return key
        
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
                content = self._read_file_with_encoding(cache_file)
                logger.info(f"Retrieved cached content for key {key}")
                return content
            
            logger.debug(f"No cached content found for key {key}")
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
            cache_file.write_text(content, encoding='utf-8')
            logger.debug(f"Content cached successfully: {key}")
            return True
        except Exception as e:
            logger.error(f"Failed to cache content: {e}")
            return False
    
    def clear_cache(self) -> bool:
        """Clear all cached content.
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            for cache_file in self.cache_dir.glob("*.md"):
                cache_file.unlink()
            logger.info("Cache cleared successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to clear cache: {e}")
            return False
    
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