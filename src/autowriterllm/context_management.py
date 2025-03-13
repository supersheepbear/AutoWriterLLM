"""Context management for AI content generation."""
from typing import Dict, List, Any, Optional
import logging
from dataclasses import dataclass, field
import time
from sentence_transformers import SentenceTransformer
import numpy as np
from pathlib import Path
import yaml

logger = logging.getLogger(__name__)

@dataclass
class ContextLevel:
    """Represents different levels of context."""
    high_level_summary: str  # Book-level overview
    recent_chapters: List[str]  # Last 2-3 chapters in detail
    concept_map: Dict[str, List[str]]  # Key concepts and their relationships
    terminology: Dict[str, str]  # Important terms and their definitions

@dataclass
class ChapterContext:
    """Stores context information for a chapter."""
    title: str
    content: str
    summary: str
    key_concepts: List[str]
    timestamp: float = field(default_factory=time.time)

class ContextManager:
    """Manages and optimizes context for AI prompts."""
    
    def __init__(self, max_context_length: int = 6000):
        self.max_context_length = max_context_length
        self.chapters: Dict[str, ChapterContext] = {}
        self.semantic_search = SemanticContextSearch()
        
    def add_chapter(self, title: str, content: str, summary: str) -> None:
        """Add a chapter to the context manager."""
        key_concepts = self._extract_key_concepts(content)
        chapter = ChapterContext(
            title=title,
            content=content,
            summary=summary,
            key_concepts=key_concepts
        )
        self.chapters[title] = chapter
        self.semantic_search.add_content(title, content)
        
    def get_optimized_context(self, current_title: str, max_tokens: int = 6000) -> str:
        """Get optimized context based on relevance and length constraints."""
        context_parts = []
        current_length = 0
        
        # Add most recent chapters first
        recent_chapters = self._get_recent_chapters(current_title, 3)
        for chapter in recent_chapters:
            if current_length + len(chapter.summary) <= max_tokens:
                context_parts.append(f"Recent Chapter {chapter.title}:\n{chapter.summary}")
                current_length += len(context_parts[-1])
        
        # Add semantically relevant content
        relevant_content = self.semantic_search.get_relevant_context(current_title)
        for content, score in relevant_content:
            if current_length + len(content) <= max_tokens:
                context_parts.append(content)
                current_length += len(content)
        
        # Add key concepts if space allows
        concepts_context = self._format_key_concepts()
        if current_length + len(concepts_context) <= max_tokens:
            context_parts.append(concepts_context)
        
        return "\n\n".join(context_parts)
    
    def _get_recent_chapters(self, current_title: str, num_chapters: int) -> List[ChapterContext]:
        """Get the most recent chapters before the current one."""
        chapter_titles = list(self.chapters.keys())
        try:
            current_idx = chapter_titles.index(current_title)
            start_idx = max(0, current_idx - num_chapters)
            return [self.chapters[title] for title in chapter_titles[start_idx:current_idx]]
        except ValueError:
            return []
    
    def _extract_key_concepts(self, content: str) -> List[str]:
        """Extract key concepts from content."""
        # Implement concept extraction logic
        # This could be enhanced with NLP techniques
        return []
    
    def _format_key_concepts(self) -> str:
        """Format key concepts into a readable context."""
        concepts = set()
        for chapter in self.chapters.values():
            concepts.update(chapter.key_concepts)
        return "Key Concepts:\n" + "\n".join(f"- {concept}" for concept in concepts)

class SemanticContextSearch:
    """Uses semantic search to find relevant context."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.content_chunks: Dict[str, List[str]] = {}
        self.embeddings: Dict[str, np.ndarray] = {}
        
    def add_content(self, title: str, content: str) -> None:
        """Add content to the semantic search index."""
        chunks = self._split_into_chunks(content)
        self.content_chunks[title] = chunks
        
        # Generate embeddings for chunks
        embeddings = self.model.encode(chunks)
        self.embeddings[title] = embeddings
        
    def get_relevant_context(self, query: str, max_chunks: int = 5) -> List[tuple[str, float]]:
        """Find most relevant content chunks for the query."""
        query_embedding = self.model.encode([query])[0]
        
        relevant_chunks = []
        for title, embeddings in self.embeddings.items():
            chunks = self.content_chunks[title]
            similarities = self._compute_similarities(query_embedding, embeddings)
            
            # Get top chunks for this title
            for idx in np.argsort(similarities)[-max_chunks:]:
                relevant_chunks.append((chunks[idx], similarities[idx]))
        
        # Sort all chunks by relevance
        relevant_chunks.sort(key=lambda x: x[1], reverse=True)
        return relevant_chunks[:max_chunks]
    
    def _split_into_chunks(self, content: str, chunk_size: int = 512) -> List[str]:
        """Split content into manageable chunks."""
        words = content.split()
        chunks = []
        current_chunk = []
        current_length = 0
        
        for word in words:
            if current_length + len(word) > chunk_size:
                chunks.append(" ".join(current_chunk))
                current_chunk = [word]
                current_length = len(word)
            else:
                current_chunk.append(word)
                current_length += len(word) + 1
        
        if current_chunk:
            chunks.append(" ".join(current_chunk))
        
        return chunks
    
    def _compute_similarities(self, query_embedding: np.ndarray, 
                            chunk_embeddings: np.ndarray) -> np.ndarray:
        """Compute cosine similarities between query and chunks."""
        return np.dot(chunk_embeddings, query_embedding) / (
            np.linalg.norm(chunk_embeddings, axis=1) * np.linalg.norm(query_embedding)
        ) 