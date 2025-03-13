from typing import Dict, List, Any
import ast
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class CodeAnalysis:
    """Results of code analysis."""
    complexity_score: float
    best_practices_score: float
    documentation_score: float
    type_hints_score: float
    suggestions: List[str]
    issues: List[str]

@dataclass
class ContentAnalysis:
    """Results of general content analysis."""
    accuracy_score: float
    clarity_score: float
    completeness_score: float
    engagement_score: float
    suggestions: List[str]
    issues: List[str]

class ContentQualityAnalyzer:
    """Analyzes and improves textbook content."""
    
    def __init__(self, ai_client: Any):
        self.ai_client = ai_client
    
    def analyze_content(self, content: str, content_type: str = "generic") -> ContentAnalysis:
        """Analyze content for quality and best practices.
        
        Args:
            content: The content to analyze
            content_type: Type of content (e.g., 'programming', 'science', 'history')
            
        Returns:
            ContentAnalysis: Analysis results
            
        Raises:
            Exception: If analysis fails
        """
        try:
            # For programming content, also analyze code blocks
            if content_type == "programming":
                code_blocks = self._extract_code_blocks(content)
                code_analysis = self._analyze_code(code_blocks)
                
                # Incorporate code analysis into content analysis
                return ContentAnalysis(
                    accuracy_score=self._analyze_accuracy(content, content_type),
                    clarity_score=self._analyze_clarity(content),
                    completeness_score=self._analyze_completeness(content, content_type),
                    engagement_score=self._analyze_engagement(content),
                    suggestions=self._generate_improvements(content, content_type) + code_analysis.suggestions,
                    issues=self._identify_issues(content, content_type) + code_analysis.issues
                )
            
            # For other content types
            return ContentAnalysis(
                accuracy_score=self._analyze_accuracy(content, content_type),
                clarity_score=self._analyze_clarity(content),
                completeness_score=self._analyze_completeness(content, content_type),
                engagement_score=self._analyze_engagement(content),
                suggestions=self._generate_improvements(content, content_type),
                issues=self._identify_issues(content, content_type)
            )
            
        except Exception as e:
            logger.error(f"Content analysis failed: {e}")
            raise
    
    def _analyze_code(self, code_blocks: List[str]) -> CodeAnalysis:
        """Analyze code blocks for quality and best practices."""
        try:
            return CodeAnalysis(
                complexity_score=self._analyze_complexity(code_blocks),
                best_practices_score=self._check_best_practices(code_blocks),
                documentation_score=self._check_documentation(code_blocks),
                type_hints_score=self._check_type_hints(code_blocks),
                suggestions=self._generate_code_improvements(code_blocks),
                issues=self._identify_code_issues(code_blocks)
            )
        except Exception as e:
            logger.error(f"Code analysis failed: {e}")
            # Return default values if analysis fails
            return CodeAnalysis(
                complexity_score=0.5,
                best_practices_score=0.5,
                documentation_score=0.5,
                type_hints_score=0.5,
                suggestions=["Code analysis failed"],
                issues=["Error analyzing code: " + str(e)]
            )
    
    def _extract_code_blocks(self, content: str) -> List[str]:
        """Extract Python code blocks from markdown content."""
        code_blocks = []
        in_block = False
        current_block = []
        
        for line in content.split('\n'):
            if line.startswith('```python'):
                in_block = True
                continue
            elif line.startswith('```') and in_block:
                in_block = False
                code_blocks.append('\n'.join(current_block))
                current_block = []
            elif in_block:
                current_block.append(line)
                
        return code_blocks
    
    def _analyze_complexity(self, code_blocks: List[str]) -> float:
        """Analyze code complexity using AST."""
        total_score = 0.0
        
        for code in code_blocks:
            try:
                tree = ast.parse(code)
                # Implement complexity analysis using AST
                # Count branches, loops, etc.
            except Exception as e:
                logger.error(f"Failed to parse code: {e}")
                
        return total_score / len(code_blocks) if code_blocks else 0.0
    
    def _check_best_practices(self, code_blocks: List[str]) -> float:
        """Check code for adherence to best practices."""
        # Implementation would check for PEP 8 compliance, etc.
        return 0.5
    
    def _check_documentation(self, code_blocks: List[str]) -> float:
        """Check code for proper documentation."""
        # Implementation would check for docstrings, comments, etc.
        return 0.5
    
    def _check_type_hints(self, code_blocks: List[str]) -> float:
        """Check code for type hints."""
        # Implementation would check for type annotations
        return 0.5
    
    def _generate_code_improvements(self, code_blocks: List[str]) -> List[str]:
        """Generate suggestions for code improvements."""
        # Implementation would suggest improvements
        return []
    
    def _identify_code_issues(self, code_blocks: List[str]) -> List[str]:
        """Identify issues in code."""
        # Implementation would identify issues
        return []
    
    def _analyze_accuracy(self, content: str, content_type: str) -> float:
        """Analyze content for factual accuracy."""
        # Implementation would analyze accuracy based on content type
        return 0.8
    
    def _analyze_clarity(self, content: str) -> float:
        """Analyze content for clarity and readability."""
        # Implementation would analyze clarity
        return 0.8
    
    def _analyze_completeness(self, content: str, content_type: str) -> float:
        """Analyze content for completeness."""
        # Implementation would analyze completeness based on content type
        return 0.8
    
    def _analyze_engagement(self, content: str) -> float:
        """Analyze content for engagement and interest level."""
        # Implementation would analyze engagement
        return 0.8
    
    def _generate_improvements(self, content: str, content_type: str) -> List[str]:
        """Generate suggestions for content improvements."""
        # Implementation would suggest improvements based on content type
        return []
    
    def _identify_issues(self, content: str, content_type: str) -> List[str]:
        """Identify issues in content."""
        # Implementation would identify issues based on content type
        return []

# Keep the original class for backward compatibility
class CodeQualityAnalyzer(ContentQualityAnalyzer):
    """Analyzes and improves code examples."""
    
    def analyze_code(self, content: str) -> CodeAnalysis:
        """Analyze code blocks for quality and best practices."""
        code_blocks = self._extract_code_blocks(content)
        return self._analyze_code(code_blocks) 