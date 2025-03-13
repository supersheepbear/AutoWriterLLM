"""Main entry point for AutoWriterLLM when run as a module.

This module allows running the AutoWriterLLM package directly with:
python -m autowriterllm
"""

import logging
import sys
from pathlib import Path

# Configure logging
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(log_dir / "autowriterllm.log")
    ]
)

logger = logging.getLogger(__name__)


def main():
    """Main entry point for the application."""
    try:
        from autowriterllm.gui import main as gui_main
        gui_main()
    except ImportError as e:
        logger.error(f"Failed to import GUI module: {e}")
        print(f"Error: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Application error: {e}")
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 