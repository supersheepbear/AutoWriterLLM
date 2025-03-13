from typing import List, Dict, Any
import logging
from dataclasses import dataclass
from autowriterllm.exceptions import GenerationError
import time

logger = logging.getLogger(__name__)

@dataclass
class QuizQuestion:
    """Represents a quiz question."""
    question: str
    options: List[str]
    correct_answer: str
    explanation: str
    type: str  # multiple_choice, true_false, code_completion, debugging

@dataclass
class Exercise:
    """Represents a practical exercise."""
    title: str
    description: str
    steps: List[str]
    difficulty: str
    expected_outcome: str
    evaluation_criteria: List[str]

class InteractiveElementsGenerator:
    """Generates interactive learning elements."""
    
    def __init__(self, ai_client: Any):
        self.ai_client = ai_client
    
    def generate_quiz(self, content: str, difficulty: str = "medium", content_type: str = "generic") -> List[QuizQuestion]:
        """Generate quiz questions from content.
        
        Args:
            content: The content to generate quiz questions from
            difficulty: Difficulty level of questions (easy, medium, hard)
            content_type: Type of content (e.g., 'programming', 'science', 'history')
            
        Returns:
            List of QuizQuestion objects
            
        Raises:
            GenerationError: If quiz generation fails
        """
        try:
            quiz_prompt = self._create_quiz_prompt(content, difficulty, content_type)
            response = self._generate_with_retry(quiz_prompt)
            return self._parse_quiz_response(response)
        except Exception as e:
            logger.error(f"Failed to generate quiz: {e}")
            raise GenerationError(f"Quiz generation failed: {str(e)}")
    
    def generate_exercises(self, content: str, level: str = "intermediate", content_type: str = "generic") -> List[Exercise]:
        """Generate practical exercises.
        
        Args:
            content: The content to generate exercises from
            level: Difficulty level (beginner, intermediate, advanced)
            content_type: Type of content (e.g., 'programming', 'science', 'history')
            
        Returns:
            List of Exercise objects
            
        Raises:
            GenerationError: If exercise generation fails
        """
        try:
            exercise_prompt = self._create_exercise_prompt(content, level, content_type)
            response = self._generate_with_retry(exercise_prompt)
            return self._parse_exercise_response(response)
        except Exception as e:
            logger.error(f"Failed to generate exercises: {e}")
            raise GenerationError(f"Exercise generation failed: {str(e)}")
    
    def _create_quiz_prompt(self, content: str, difficulty: str, content_type: str = "generic") -> str:
        """Create prompt for quiz generation.
        
        Args:
            content: The content to generate quiz questions from
            difficulty: Difficulty level of questions
            content_type: Type of content
            
        Returns:
            str: The quiz generation prompt
        """
        base_prompt = f"""Create varied quiz questions from this {content_type} content:
{content[:2000]}

Generate diverse question types appropriate for {difficulty} difficulty level.
Include detailed explanations for correct answers.

Format response as YAML with questions list containing:
- question: string
- options: list of strings
- correct_answer: string
- explanation: string
- type: string (multiple_choice/true_false/short_answer/matching)"""

        # Add content-specific quiz instructions
        if content_type == "programming":
            return base_prompt + """

For programming content, include:
1. Multiple choice questions about concepts
2. True/False questions about best practices
3. Code completion exercises
4. Debugging challenges"""
        elif content_type == "science":
            return base_prompt + """

For science content, include:
1. Multiple choice questions about scientific concepts
2. True/False questions about experimental results
3. Hypothesis formation exercises
4. Data interpretation questions"""
        elif content_type == "mathematics":
            return base_prompt + """

For mathematics content, include:
1. Multiple choice questions about mathematical concepts
2. True/False questions about theorems
3. Problem-solving exercises
4. Proof completion questions"""
        elif content_type == "history":
            return base_prompt + """

For history content, include:
1. Multiple choice questions about historical events
2. True/False questions about historical figures
3. Timeline ordering exercises
4. Primary source analysis questions"""
        elif content_type == "literature":
            return base_prompt + """

For literature content, include:
1. Multiple choice questions about literary works
2. True/False questions about authors
3. Text analysis exercises
4. Character identification questions"""
        elif content_type == "business":
            return base_prompt + """

For business content, include:
1. Multiple choice questions about business concepts
2. True/False questions about market trends
3. Case study analysis exercises
4. Strategy development questions"""
        elif content_type == "arts":
            return base_prompt + """

For arts content, include:
1. Multiple choice questions about artistic movements
2. True/False questions about artists
3. Style identification exercises
4. Technique application questions"""
        elif content_type == "medicine":
            return base_prompt + """

For medicine content, include:
1. Multiple choice questions about medical concepts
2. True/False questions about treatments
3. Diagnostic reasoning exercises
4. Patient case analysis questions"""
        elif content_type == "philosophy":
            return base_prompt + """

For philosophy content, include:
1. Multiple choice questions about philosophical concepts
2. True/False questions about philosophers
3. Argument analysis exercises
4. Ethical reasoning questions"""
        elif content_type == "language":
            return base_prompt + """

For language content, include:
1. Multiple choice questions about grammar rules
2. True/False questions about usage
3. Translation exercises
4. Sentence construction questions"""
        else:
            return base_prompt
    
    def _create_exercise_prompt(self, content: str, level: str, content_type: str = "generic") -> str:
        """Create prompt for exercise generation.
        
        Args:
            content: The content to generate exercises from
            level: Difficulty level
            content_type: Type of content
            
        Returns:
            str: The exercise generation prompt
        """
        base_prompt = f"""Create practical exercises for this {content_type} content:
{content[:2000]}

Generate {level} level exercises that help readers apply what they've learned.
Include clear instructions, steps, and evaluation criteria.

Format response as YAML with exercises list containing:
- title: string
- description: string
- steps: list of strings
- difficulty: string
- expected_outcome: string
- evaluation_criteria: list of strings"""

        # Add content-specific exercise instructions
        if content_type == "programming":
            return base_prompt + """

For programming exercises, include:
1. Code implementation tasks
2. Debugging exercises
3. Algorithm optimization challenges
4. Project-based applications"""
        elif content_type == "science":
            return base_prompt + """

For science exercises, include:
1. Experimental design tasks
2. Data analysis exercises
3. Hypothesis testing challenges
4. Research proposal development"""
        elif content_type == "mathematics":
            return base_prompt + """

For mathematics exercises, include:
1. Problem-solving tasks
2. Proof development exercises
3. Real-world application challenges
4. Mathematical modeling activities"""
        elif content_type == "history":
            return base_prompt + """

For history exercises, include:
1. Primary source analysis tasks
2. Historical argument development exercises
3. Comparative history challenges
4. Historical research activities"""
        elif content_type == "literature":
            return base_prompt + """

For literature exercises, include:
1. Text analysis tasks
2. Creative writing exercises
3. Literary criticism challenges
4. Comparative literature activities"""
        elif content_type == "business":
            return base_prompt + """

For business exercises, include:
1. Case study analysis tasks
2. Business plan development exercises
3. Market analysis challenges
4. Strategic decision-making activities"""
        elif content_type == "arts":
            return base_prompt + """

For arts exercises, include:
1. Technique application tasks
2. Style analysis exercises
3. Creative production challenges
4. Art criticism activities"""
        elif content_type == "medicine":
            return base_prompt + """

For medicine exercises, include:
1. Diagnostic reasoning tasks
2. Treatment planning exercises
3. Patient case analysis challenges
4. Medical research evaluation activities"""
        elif content_type == "philosophy":
            return base_prompt + """

For philosophy exercises, include:
1. Argument analysis tasks
2. Ethical reasoning exercises
3. Philosophical writing challenges
4. Thought experiment development activities"""
        elif content_type == "language":
            return base_prompt + """

For language exercises, include:
1. Grammar application tasks
2. Translation exercises
3. Conversation practice challenges
4. Writing and composition activities"""
        else:
            return base_prompt
    
    def _generate_with_retry(self, prompt: str, max_retries: int = 3) -> str:
        """Generate content with retry logic.
        
        Args:
            prompt: The prompt to send to the AI
            max_retries: Maximum number of retry attempts
            
        Returns:
            str: The generated content
            
        Raises:
            GenerationError: If generation fails after max retries
        """
        for attempt in range(max_retries):
            try:
                response = self.ai_client.messages.create(
                    model="claude-3-sonnet-20240229",
                    max_tokens=2000,
                    temperature=0.7,
                    messages=[{"role": "user", "content": prompt}]
                )
                return response.content[0].text
            except Exception as e:
                logger.warning(f"Generation attempt {attempt+1} failed: {e}")
                if attempt == max_retries - 1:
                    raise GenerationError(f"Failed after {max_retries} attempts: {str(e)}")
                time.sleep(2 ** attempt)  # Exponential backoff
    
    def _parse_quiz_response(self, response: str) -> List[QuizQuestion]:
        """Parse quiz response from AI.
        
        Args:
            response: YAML-formatted response from AI
            
        Returns:
            List of QuizQuestion objects
            
        Raises:
            GenerationError: If parsing fails
        """
        try:
            # Implementation would parse YAML and create QuizQuestion objects
            # This is a placeholder
            return []
        except Exception as e:
            logger.error(f"Failed to parse quiz response: {e}")
            raise GenerationError(f"Quiz parsing failed: {str(e)}")
    
    def _parse_exercise_response(self, response: str) -> List[Exercise]:
        """Parse exercise response from AI.
        
        Args:
            response: YAML-formatted response from AI
            
        Returns:
            List of Exercise objects
            
        Raises:
            GenerationError: If parsing fails
        """
        try:
            # Implementation would parse YAML and create Exercise objects
            # This is a placeholder
            return []
        except Exception as e:
            logger.error(f"Failed to parse exercise response: {e}")
            raise GenerationError(f"Exercise parsing failed: {str(e)}") 