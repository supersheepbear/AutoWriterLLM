=====
Usage
=====

Basic Usage
----------

AutoWriterLLM provides both GUI and programmatic interfaces for generating book content:

GUI Interface
^^^^^^^^^^^^

Launch the GUI application:

.. code-block:: python

    from autowriterllm.ai_content_generator import ContentGenerator

    # Launch the GUI
    generator = ContentGenerator()
    generator.run_gui()

Programmatic Interface
^^^^^^^^^^^^^^^^^^^^

Generate content programmatically:

.. code-block:: python

    from autowriterllm.ai_content_generator import ContentGenerator
    import logging

    # Configure logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # Initialize generator with configuration
    generator = ContentGenerator(config_file="config.yaml")

    try:
        # Generate book content
        generator.generate_book_plan(
            topic="Python Programming",
            level="Intermediate",
            provider="anthropic"  # or "lingyiwanwu"
        )
    except Exception as e:
        logger.error(f"Error generating book plan: {e}")

Configuration
------------

Create a ``config.yaml`` file:

.. code-block:: yaml

    # Anthropic Configuration
    anthropic:
      api_key: your-anthropic-api-key-here
      model: claude-3-sonnet-20240229

    # Lingyiwanwu Configuration
    lingyiwanwu:
      api_key: your-lingyiwanwu-api-key-here
      model: yi-lightning

    # Default Provider
    provider: anthropic

    # Output Configuration
    output:
      directory: "output"
      format: "markdown"

    # Logging Configuration
    logging:
      level: "INFO"
      file: "logs/autowriter.log"

PDF Conversion
-------------

Convert markdown content to PDF:

.. code-block:: python

    from autowriterllm.markdown_to_pdf_converter import MarkdownToPDFConverter

    converter = MarkdownToPDFConverter()
    
    # Convert single file
    converter.convert(
        input_file="output/chapter-1.md",
        output_file="output/chapter-1.pdf"
    )

    # Convert entire book
    converter.convert_book(
        toc_file="output/table_of_contents.md",
        input_dir="output",
        output_file="output/book.pdf"
    )

Error Handling
-------------

The package includes comprehensive error handling and logging:

.. code-block:: python

    import logging
    from autowriterllm.ai_content_generator import ContentGenerator
    from pathlib import Path

    # Configure logging
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler("logs/content_generation.log"),
            logging.StreamHandler()
        ]
    )

    try:
        generator = ContentGenerator()
        generator.generate_book_plan(topic="FastAPI Development")
    except Exception as e:
        logging.error(f"Book generation failed: {e}")
