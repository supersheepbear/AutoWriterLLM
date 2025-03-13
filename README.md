# AutoWriterLLM

A Python tool that helps automatically generate comprehensive book structures and content using Large Language Models (LLMs). It provides functionality for content generation, markdown to PDF conversion, and interactive content creation.

## Features

- **Universal Textbook Generation**: Generate content for any type of textbook, including programming, science, mathematics, history, literature, business, arts, medicine, philosophy, and language.
- **Multi-Part Content Generation**: Break down each chapter into multiple parts for more comprehensive, detailed content with better examples and explanations.
- **Content Generation**: Automatically generate comprehensive book structures and content using LLMs.
- **Markdown to PDF Conversion**: Convert generated markdown content to PDF format with customizable styling.
- **Interactive Content Creation**: Generate quizzes, exercises, and other interactive elements tailored to the content type.
- **Content Review**: AI-powered review system to ensure quality and accuracy of generated content.
- **Learning Path Optimization**: Generate customized learning paths based on content and user preferences.

## Installation

```bash
pip install autowriterllm
```

## Usage

### Command Line Interface

```bash
# Generate a book structure
autowriterllm plan "Create an intermediate history textbook on Ancient Rome"

# Generate content from a table of contents
autowriterllm generate --toc table_of_contents.md --output ./book_content/

# Convert markdown to PDF
autowriterllm convert --input ./book_content/ --output book.pdf
```

### Python API

```python
from autowriterllm import ContentGenerator

# Initialize the generator
generator = ContentGenerator("table_of_contents.md", "output_directory")

# Parse the table of contents
generator.parse_toc()

# Generate all content
generator.generate_all_content()
```

## Content Types

The system automatically detects the type of content being generated and tailors the output accordingly. Supported content types include:

- **Programming**: Code examples, algorithms, best practices
- **Science**: Experiments, theories, research methods
- **Mathematics**: Formulas, proofs, problem-solving
- **History**: Timelines, events, historical analysis
- **Literature**: Literary analysis, author studies, text interpretation
- **Business**: Case studies, frameworks, market analysis
- **Arts**: Techniques, movements, artistic analysis
- **Medicine**: Diagnoses, treatments, medical research
- **Philosophy**: Concepts, arguments, ethical reasoning
- **Language**: Grammar, vocabulary, linguistic analysis

## Configuration

Create a `config.yaml` file to customize the behavior:

```yaml
# API Provider Configuration
provider: "anthropic"
anthropic:
  api_key: "your_anthropic_api_key_here"
  model: "claude-3-opus-20240229"

# Content Generation Options
multi_part_generation: true  # Enable multi-part content generation
parts_per_section: 5         # Number of parts per section
max_tokens_per_part: 8192    # Maximum tokens per part

# Context Management Options
include_learning_metadata: true        # Include learning objectives
include_previous_chapter_context: true # Include context from previous chapters
include_previous_section_context: true # Include context from previous sections

# Output Options
output_dir: "output"  # Directory for generated content
```

### Multi-Part Content Generation

The multi-part content generation feature breaks down each chapter into multiple parts, resulting in more comprehensive and detailed content. This approach:

1. **Increases Content Depth**: Each part focuses on different aspects of the topic, allowing for more thorough coverage.
2. **Improves Examples**: More space is dedicated to providing multiple, detailed examples.
3. **Enhances Explanations**: Concepts are explained more thoroughly with different approaches.
4. **Eliminates Recaps**: Content flows naturally without unnecessary recaps of previous chapters.
5. **Removes Redundant Summaries**: The content focuses on educational material rather than summarizing what was just covered.

You can configure this feature in the `config.yaml` file:

```yaml
multi_part_generation: true  # Enable/disable multi-part generation
parts_per_section: 5         # Number of parts to generate per section
max_tokens_per_part: 8192    # Maximum tokens per part
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details. 