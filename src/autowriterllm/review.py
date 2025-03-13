from dataclasses import dataclass
from typing import List, Optional, Dict, Any
import logging
import yaml
from autowriterllm.exceptions import ReviewError

logger = logging.getLogger(__name__)

@dataclass
class ReviewFeedback:
    """Stores feedback from content review."""
    section_title: str
    technical_accuracy: float  # 0-1 score
    clarity: float
    completeness: float
    comments: List[str]
    suggested_improvements: List[str]

class ContentReviewer:
    """AI-powered content review system."""
    
    def __init__(self, ai_client: Any):
        """Initialize reviewer with AI client."""
        self.ai_client = ai_client
    
    def review_content(self, content: str, section_title: str, content_type: str = "generic") -> ReviewFeedback:
        """Perform comprehensive content review.
        
        Args:
            content: The content to review
            section_title: Title of the section being reviewed
            content_type: Type of content (e.g., 'programming', 'science', 'history')
            
        Returns:
            ReviewFeedback object containing the review results
            
        Raises:
            ReviewError: If review process fails
        """
        try:
            review_prompt = self._create_review_prompt(content, section_title, content_type)
            review_yaml = self._generate_review(review_prompt)
            review_data = yaml.safe_load(review_yaml)
            
            # Validate review data
            required_fields = ['technical_accuracy', 'clarity', 'completeness', 
                             'comments', 'suggested_improvements']
            for field in required_fields:
                if field not in review_data:
                    raise ReviewError(f"Missing required field in review: {field}")
            
            return ReviewFeedback(
                section_title=section_title,
                **review_data
            )
            
        except Exception as e:
            logger.error(f"Review failed for {section_title}: {e}")
            raise ReviewError(f"Review failed: {str(e)}")
    
    def _create_review_prompt(self, content: str, section_title: str, content_type: str = "generic") -> str:
        """Create prompt for content review.
        
        Args:
            content: The content to review
            section_title: Title of the section being reviewed
            content_type: Type of content (e.g., 'programming', 'science', 'history')
            
        Returns:
            str: The review prompt
        """
        base_prompt = f"""Review this {content_type} textbook content and provide detailed feedback:

Section: {section_title}

Content:
{content[:3000]}

Evaluate and score (0-1) the following aspects:
1. Technical accuracy: Are concepts correctly explained? Is the information accurate?
2. Clarity: Is the content clear and well-structured?
3. Completeness: Are all necessary topics covered?

Also provide:
1. Specific comments on strengths and weaknesses
2. Concrete suggestions for improvement

Format your response as YAML with these keys:
- technical_accuracy: float
- clarity: float
- completeness: float
- comments: list of strings
- suggested_improvements: list of strings"""

        # Add content-specific review criteria
        if content_type == "programming":
            return base_prompt + """

For programming content, also consider:
- Code correctness and best practices
- Error handling and edge cases
- Documentation quality
- Performance considerations"""
        elif content_type == "science":
            return base_prompt + """

For science content, also consider:
- Scientific accuracy and current understanding
- Experimental methodology descriptions
- Data interpretation
- Citation of relevant research"""
        elif content_type == "mathematics":
            return base_prompt + """

For mathematics content, also consider:
- Mathematical notation accuracy
- Proof correctness and clarity
- Step-by-step explanations
- Visual representations of concepts"""
        elif content_type == "history":
            return base_prompt + """

For history content, also consider:
- Historical accuracy and context
- Multiple perspectives and interpretations
- Chronological clarity
- Primary and secondary source usage"""
        elif content_type == "literature":
            return base_prompt + """

For literature content, also consider:
- Literary analysis depth
- Textual evidence usage
- Cultural and historical context
- Author background information"""
        elif content_type == "business":
            return base_prompt + """

For business content, also consider:
- Case study relevance and analysis
- Framework and model explanations
- Market trend accuracy
- Practical application guidance"""
        elif content_type == "arts":
            return base_prompt + """

For arts content, also consider:
- Technique descriptions
- Historical context of movements
- Analysis depth of works
- Visual descriptions and examples"""
        elif content_type == "medicine":
            return base_prompt + """

For medicine content, also consider:
- Medical accuracy and current practices
- Anatomical descriptions
- Treatment protocol explanations
- Evidence-based recommendations"""
        elif content_type == "philosophy":
            return base_prompt + """

For philosophy content, also consider:
- Philosophical concept explanations
- Argument structure and logic
- Historical context of ideas
- Multiple perspective presentation"""
        elif content_type == "language":
            return base_prompt + """

For language content, also consider:
- Grammatical accuracy
- Example clarity and relevance
- Cultural context explanations
- Pronunciation guidance"""
        else:
            return base_prompt
    
    def _generate_review(self, prompt: str) -> str:
        """Generate review using AI client."""
        # Implementation depends on AI client type
        response = self.ai_client.messages.create(
            model="claude-3-sonnet-20240229",
            max_tokens=2000,
            temperature=0.7,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.content[0].text 