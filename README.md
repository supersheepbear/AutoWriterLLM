# AutoWriterLLM

A Python tool that helps automatically generate comprehensive book structures and content using Large Language Models (LLMs). It provides functionality for content generation, markdown to PDF conversion, and interactive content creation.

## Features

- **Universal Textbook Generation**: Generate content for any type of textbook, including programming, science, mathematics, history, literature, business, arts, medicine, philosophy, and language.
- **Multi-Part Content Generation**: Break down each chapter into multiple parts for more comprehensive, detailed content with better examples and explanations.
- **Flexible LLM Integration**: Support for any LLM provider with customizable models and API endpoints.
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

Create a `config.yaml` file to customize the behavior. The new configuration structure supports any LLM provider:

```yaml
# Default Provider
provider: "anthropic"  # The provider to use by default

# Provider Configurations
providers:
  # Anthropic Configuration
  anthropic:
    api_key: "your_anthropic_api_key_here"
    base_url: "https://api.anthropic.com"
    models:
      - name: "claude-3-opus-20240229"
        description: "Most powerful Claude model with advanced reasoning"
        max_tokens: 8192
        temperature: 0.7
      - name: "claude-3-sonnet-20240229"
        description: "Balanced Claude model for most tasks"
        max_tokens: 8192
        temperature: 0.7
  
  # OpenAI Configuration
  openai:
    api_key: "your_openai_api_key_here"
    base_url: "https://api.openai.com/v1"
    models:
      - name: "gpt-4o"
        description: "Latest GPT-4 Omni model"
        max_tokens: 8192
        temperature: 0.7
      - name: "gpt-3.5-turbo"
        description: "Efficient GPT-3.5 model"
        max_tokens: 4096
        temperature: 0.7
  
  # Custom Provider Example
  custom_provider:
    api_key: "your_api_key_here"
    base_url: "https://api.custom-provider.com/v1"
    models:
      - name: "model-name"
        description: "Model description"
        max_tokens: 4096
        temperature: 0.7

# Content Generation Options
multi_part_generation: true  # Enable multi-part content generation
parts_per_section: 5         # Number of parts per section
max_tokens_per_part: 8192    # Maximum tokens per part
```

### Adding Custom LLM Providers

You can add any LLM provider by following these steps:

1. Add a new provider section to the `providers` configuration in `config.yaml`
2. Specify the `api_key` and `base_url` for the provider
3. Define the available models with their parameters
4. Set the provider as the default by updating the `provider` field

The system will automatically handle the API calls to the provider using the specified configuration.

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