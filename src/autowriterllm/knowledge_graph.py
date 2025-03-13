"""Knowledge graph implementation for tracking concepts."""
from typing import List, Dict, Optional
from dataclasses import dataclass

@dataclass
class ConceptNode:
    """Represents a concept in the knowledge graph."""
    name: str
    description: str
    prerequisites: List[str]
    related_concepts: List[str]

class KnowledgeGraph:
    """Maintains relationships between concepts across chapters."""
    def __init__(self):
        self.concepts: Dict[str, ConceptNode] = {}
        
    def add_concept(self, concept: ConceptNode) -> None:
        """Add a concept to the graph."""
        self.concepts[concept.name] = concept
        
    def get_related_concepts(self, concept_name: str) -> List[str]:
        """Get related concepts for a given concept."""
        return self.concepts[concept_name].related_concepts if concept_name in self.concepts else []
    
    def get_concepts_for_section(self, section_title: str) -> List[ConceptNode]:
        """Get concepts related to a section."""
        # Implementation depends on how you want to map sections to concepts
        return [concept for concept in self.concepts.values()] 