=====
Usage
=====

Basic Usage
----------

AutoWriterLLM can be used either through its GUI interface or programmatically.

GUI Interface
^^^^^^^^^^^^

To launch the GUI interface::

    from autowriterllm.markdown_to_pdf_converter import ConverterGUI
    
    gui = ConverterGUI()
    gui.run()

The GUI provides easy access to all conversion features:

1. Select input markdown files
2. Choose output PDF location
3. Configure conversion options
4. Monitor conversion progress

Programmatic Interface
^^^^^^^^^^^^^^^^^^^^

For programmatic usage::

    from autowriterllm.markdown_to_pdf_converter import MarkdownToPDFConverter
    
    converter = MarkdownToPDFConverter()
    
    # Convert single file
    converter.convert_to_pdf(
        input_file="path/to/input.md",
        output_file="path/to/output.pdf"
    )
    
    # Convert entire book
    converter.convert_book(
        toc_file="path/to/toc.md",
        input_dir="path/to/chapters",
        output_file="path/to/book.pdf"
    )

Configuration
------------

Create a ``config.yaml`` file in your project root::

    # API Configuration
    api:
      anthropic:
        api_key: "your-api-key"
        model: "claude-3-sonnet-20240229"
        max_tokens: 4096
        temperature: 0.7

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

The PDF converter supports various options::

    converter = MarkdownToPDFConverter(
        config={
            "pandoc_options": [
                "--pdf-engine=xelatex",
                "--toc",
                "--number-sections"
            ],
            "supported_encodings": ["utf-8", "cp1252"],
            "max_workers": 4
        }
    )

Error Handling
-------------

The library includes comprehensive error handling::

    try:
        converter.convert_to_pdf(input_file="input.md", output_file="output.pdf")
    except FileNotFoundError:
        logger.error("Input file not found")
    except subprocess.CalledProcessError as e:
        logger.error(f"Conversion failed: {e}")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")

For more detailed error handling, enable debug logging::

    import logging
    logging.basicConfig(level=logging.DEBUG)
