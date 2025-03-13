# AutoWriterLLM

A Python tool that helps automatically generate comprehensive book structures and content using Large Language Models (LLMs). It provides functionality for content generation, markdown to PDF conversion, and interactive content creation.

## Features

- **Universal Textbook Generation**: Generate content for any type of textbook, including programming, science, mathematics, history, literature, business, arts, medicine, philosophy, and language.
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
provider: anthropic
anthropic:
  api_key: your_api_key
  model: claude-3-sonnet-20240229
output:
  format: markdown
  style: academic
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details. 