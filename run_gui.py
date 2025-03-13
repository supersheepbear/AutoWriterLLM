#!/usr/bin/env python
"""
Simple script to launch the AutoWriterLLM GUI.

Usage:
    python run_gui.py
"""

import sys
import logging
from pathlib import Path

# Add src directory to path if running from project root
src_path = Path(__file__).parent / "src"
if src_path.exists():
    sys.path.insert(0, str(src_path))

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

if __name__ == "__main__":
    try:
        from autowriterllm.gui import main
        main()
    except ImportError as e:
        logger.error(f"Failed to import GUI module: {e}")
        print(f"Error: {e}")
        print("Make sure you have installed the package or are running from the project root.")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Application error: {e}")
        print(f"Error: {e}")
        sys.exit(1) 