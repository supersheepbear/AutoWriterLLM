import pytest
from unittest.mock import MagicMock, patch
from src.autowriterllm.learning_path import LearningPathOptimizer, LearningObjective
from src.autowriterllm.knowledge_graph import KnowledgeGraph

@pytest.fixture
def mock_knowledge_graph():
    """Mock knowledge graph for testing."""
    graph = MagicMock(spec=KnowledgeGraph)
    
    # Mock concept data
    concept1 = MagicMock()
    concept1.name = "Concept 1"
    concept1.prerequisites = ["Prerequisite 1"]
    
    concept2 = MagicMock()
    concept2.name = "Algorithm"
    concept2.prerequisites = ["Prerequisite 2"]
    
    # Mock get_concepts_for_section to return different concepts based on section
    def get_concepts_for_section(title):
        if "algorithms" in title.lower():
            return [concept2]
        return [concept1]
    
    graph.get_concepts_for_section = get_concepts_for_section
    
    return graph

@pytest.fixture
def mock_section():
    """Mock section for testing."""
    section = MagicMock()
    section.title = "Test Section"
    return section

def test_analyze_progression_with_content_type(mock_knowledge_graph):
    """Test analyzing progression with different content types."""
    optimizer = LearningPathOptimizer(mock_knowledge_graph)
    
    # Mock internal methods
    optimizer._analyze_complexity_progression = MagicMock(return_value={"complexity": "data"})
    optimizer._analyze_dependencies = MagicMock(return_value={"dependencies": "data"})
    optimizer._identify_gaps = MagicMock(return_value=["gap1", "gap2"])
    optimizer._generate_suggestions = MagicMock(return_value=["suggestion1", "suggestion2"])
    
    # Test with programming content type
    sections = [MagicMock()]
    result = optimizer.analyze_progression(sections, "programming")
    
    # Verify the result structure
    assert "complexity_curve" in result
    assert "concept_dependencies" in result
    assert "knowledge_gaps" in result
    assert "suggested_improvements" in result
    
    # Verify method calls with correct content type
    optimizer._analyze_complexity_progression.assert_called_with(sections, "programming")
    optimizer._identify_gaps.assert_called_with(sections, "programming")
    optimizer._generate_suggestions.assert_called_with(sections, "programming")

def test_suggest_prerequisites_with_content_type(mock_knowledge_graph, mock_section):
    """Test suggesting prerequisites with different content types."""
    optimizer = LearningPathOptimizer(mock_knowledge_graph)
    
    # Test with programming content type
    mock_section.title = "Algorithms and Data Structures"
    prerequisites = optimizer.suggest_prerequisites(mock_section, "programming")
    
    # Verify that programming-specific prerequisites are included
    assert "Basic algorithm understanding" in prerequisites
    assert "Prerequisite 2" in prerequisites
    
    # Test with mathematics content type
    mock_section.title = "Calculus Concepts"
    prerequisites = optimizer.suggest_prerequisites(mock_section, "mathematics")
    
    # Verify that mathematics-specific prerequisites might be included
    # (depending on the mock implementation)
    assert len(prerequisites) > 0

def test_generate_learning_objectives_with_content_type(mock_knowledge_graph, mock_section):
    """Test generating learning objectives with different content types."""
    optimizer = LearningPathOptimizer(mock_knowledge_graph)
    
    # Test with programming content type
    objectives = optimizer.generate_learning_objectives(mock_section, "programming")
    
    # Verify that programming-specific objectives are included
    programming_objectives = [obj for obj in objectives 
                             if "programming" in str(obj.description).lower() or 
                                "code" in str(obj.evaluation_criteria).lower()]
    assert len(programming_objectives) > 0
    
    # Test with science content type
    objectives = optimizer.generate_learning_objectives(mock_section, "science")
    
    # Verify that science-specific objectives are included
    science_objectives = [obj for obj in objectives 
                         if "experiment" in str(obj.description).lower() or 
                            "scientific" in str(obj.evaluation_criteria).lower()]
    assert len(science_objectives) > 0

def test_create_objective_for_concept_with_content_type(mock_knowledge_graph):
    """Test creating objectives for concepts with different content types."""
    optimizer = LearningPathOptimizer(mock_knowledge_graph)
    
    # Create a test concept
    concept = MagicMock()
    concept.name = "Test Concept"
    concept.prerequisites = ["Prerequisite 1"]
    
    # Test with programming content type
    objective = optimizer._create_objective_for_concept(concept, "programming")
    
    # Verify the objective structure
    assert isinstance(objective, LearningObjective)
    assert "Test Concept" in objective.description
    assert objective.prerequisites == ["Prerequisite 1"]
    
    # Verify that programming-specific criteria are included
    code_criteria = [crit for crit in objective.evaluation_criteria 
                    if "code" in crit.lower() or "implement" in crit.lower()]
    assert len(code_criteria) > 0
    
    # Test with mathematics content type
    objective = optimizer._create_objective_for_concept(concept, "mathematics")
    
    # Verify that mathematics-specific criteria are included
    math_criteria = [crit for crit in objective.evaluation_criteria 
                    if "problem" in crit.lower() or "solve" in crit.lower()]
    assert len(math_criteria) > 0

def test_content_specific_learning_objectives(mock_knowledge_graph, mock_section):
    """Test that content-specific learning objectives are added."""
    optimizer = LearningPathOptimizer(mock_knowledge_graph)
    
    # Mock get_concepts_for_section to return empty list to focus on content-specific objectives
    mock_knowledge_graph.get_concepts_for_section.return_value = []
    
    # Test with different content types
    programming_objectives = optimizer.generate_learning_objectives(mock_section, "programming")
    mathematics_objectives = optimizer.generate_learning_objectives(mock_section, "mathematics")
    science_objectives = optimizer.generate_learning_objectives(mock_section, "science")
    history_objectives = optimizer.generate_learning_objectives(mock_section, "history")
    literature_objectives = optimizer.generate_learning_objectives(mock_section, "literature")
    
    # Verify that each content type has specific objectives
    assert any("programming" in str(obj.description).lower() for obj in programming_objectives)
    assert any("mathematical" in str(obj.description).lower() for obj in mathematics_objectives)
    assert any("scientific" in str(obj.description).lower() for obj in science_objectives)
    assert any("historical" in str(obj.description).lower() for obj in history_objectives)
    assert any("literary" in str(obj.description).lower() for obj in literature_objectives) 