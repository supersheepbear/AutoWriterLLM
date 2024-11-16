import logging
import tkinter as tk
from tkinter import ttk, filedialog, scrolledtext
from pathlib import Path
from typing import Optional, List, Dict, Any
import datetime
import threading
import re
import subprocess
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
import queue

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("pdf_conversion.log"), logging.StreamHandler()],
)

logger = logging.getLogger(__name__)


@dataclass
class ConverterConfig:
    """Configuration settings for the PDF converter.
    
    Attributes:
        max_workers (int): Maximum number of worker threads for parallel processing
        supported_encodings (List[str]): List of encodings to try when reading files
        temp_filename (str): Name of temporary combined markdown file
        pandoc_options (Dict[str, Any]): Additional pandoc conversion options
    """
    max_workers: int = 4
    supported_encodings: List[str] = field(default_factory=lambda: [
        "utf-8", "utf-8-sig", "gbk", "gb2312", "gb18030", "latin1"
    ])
    temp_filename: str = "combined_temp.md"
    pandoc_options: Dict[str, Any] = field(default_factory=lambda: {
        "toc_depth": 3,
        "margin": "1in",
        "highlight_style": "tango"
    })


class MarkdownToPDFConverter:
    """Converts markdown files to a single PDF document using pandoc."""

    def __init__(
        self, toc_file: Path, output_dir: Path, css_file: Optional[Path] = None,
        config: Optional[ConverterConfig] = None
    ):
        """Initialize the converter.

        Args:
            toc_file: Path to table of contents markdown file
            output_dir: Directory containing markdown files
            css_file: Optional path to custom CSS file
            config: Optional configuration settings

        Raises:
            ValueError: If toc_file or output_dir don't exist
        """
        self.toc_file = Path(toc_file)
        self.output_dir = Path(output_dir)
        self.css_file = css_file
        self.config = config or ConverterConfig()
        
        # Validate paths
        if not self.toc_file.exists():
            raise ValueError(f"TOC file not found: {self.toc_file}")
        if not self.output_dir.is_dir():
            raise ValueError(f"Invalid output directory: {self.output_dir}")
            
        # Initialize thread-safe queue for logging
        self.log_queue = queue.Queue()

    def _check_dependencies(self) -> None:
        """Check if required dependencies are installed."""
        try:
            # Check pandoc
            result = subprocess.run(
                ["pandoc", "--version"], capture_output=True, text=True
            )
            logger.info("Found pandoc installation")

            # Check wkhtmltopdf - try multiple possible locations
            wkhtmltopdf_paths = [
                "wkhtmltopdf",  # If in PATH
                r"C:\Program Files\wkhtmltopdf\bin\wkhtmltopdf.exe",
                r"C:\Program Files (x86)\wkhtmltopdf\bin\wkhtmltopdf.exe",
            ]

            for path in wkhtmltopdf_paths:
                try:
                    result = subprocess.run(
                        [path, "--version"], capture_output=True, text=True
                    )
                    self.wkhtmltopdf_path = path  # Store the working path
                    logger.info(f"Found wkhtmltopdf at: {path}")
                    break
                except FileNotFoundError:
                    continue
            else:
                raise FileNotFoundError(
                    "wkhtmltopdf not found. Please install wkhtmltopdf from: "
                    "https://wkhtmltopdf.org/downloads.html"
                )

        except FileNotFoundError as e:
            if "pandoc" in str(e):
                raise RuntimeError("pandoc not found. Please install pandoc first.")
            raise

    def _parse_toc(self) -> List[str]:
        """Parse table of contents to get ordered list of markdown files.
        
        Returns:
            List[str]: Ordered list of markdown filenames
            
        Example ordering:
            chapter-1.md
            chapter-1-1.md
            chapter-1-1-1.md
            chapter-1-2.md
            chapter-2.md
            chapter-2-1.md
        """
        try:
            logger.info("Parsing table of contents...")

            # Get all markdown files that start with "chapter-" in the directory
            all_md_files = [
                f.name for f in self.output_dir.glob("chapter-*.md")
            ]
            logger.info(f"Found {len(all_md_files)} chapter files in directory")

            if not all_md_files:
                logger.warning("No chapter files found in directory")
                return []

            # Custom sorting function for chapter files
            def chapter_sort_key(filename: str) -> tuple:
                """Create a sort key for chapter filenames.
                
                Args:
                    filename: The filename to create a sort key for
                    
                Returns:
                    A tuple of integers representing the chapter number
                    
                Example:
                    'chapter-1-2.md' -> (1, 2)
                    'chapter-1. introduction.md' -> (1,)
                """
                # Remove 'chapter-' prefix and '.md' suffix
                base = filename.replace('chapter-', '').replace('.md', '')
                
                # Extract all numbers from the start of the string
                numbers = []
                for part in base.split('-'):
                    # Extract first number from the part
                    match = re.match(r'(\d+)', part)
                    if match:
                        numbers.append(int(match.group(1)))
                    else:
                        break
                
                return tuple(numbers) if numbers else (float('inf'),)  # Return inf if no numbers found

            # Sort files using the custom sort key
            sorted_files = sorted(all_md_files, key=chapter_sort_key)
            
            logger.info("Sorted chapter files in correct order:")
            for f in sorted_files:
                logger.debug(f"  {f}")

            return sorted_files

        except Exception as e:
            logger.error(f"Error parsing table of contents: {e}")
            raise

    def _read_file_with_fallback_encoding(self, file_path: Path) -> str:
        """Read file content with fallback encodings."""
        encodings = ["utf-8-sig", "utf-8", "latin1"]  # Simplified encoding list
        
        errors = []
        for encoding in encodings:
            try:
                logger.debug(f"Trying to read {file_path} with {encoding} encoding")
                with open(file_path, 'r', encoding=encoding) as f:
                    content = f.read()
                    # Normalize line endings
                    content = content.replace('\r\n', '\n')
                    return content
            except UnicodeError as e:
                errors.append(f"{encoding}: {str(e)}")
                continue

        error_msg = f"Failed to read {file_path} with any supported encoding:\n" + "\n".join(errors)
        logger.error(error_msg)
        raise UnicodeError(error_msg)

    def _process_markdown_files(self, files: List[str]) -> List[str]:
        """Process markdown files in parallel.
        
        Args:
            files: List of markdown filenames to process
            
        Returns:
            List[str]: Processed markdown content
            
        Raises:
            RuntimeError: If any file processing fails
        """
        processed_content = []
        errors = []

        def process_file(filename: str) -> str:
            try:
                file_path = self.output_dir / filename
                content = self._read_file_with_fallback_encoding(file_path)
                self.log_queue.put(f"Processed {filename}")
                return content
            except Exception as e:
                errors.append((filename, str(e)))
                return ""

        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            processed_content = list(executor.map(process_file, files))

        if errors:
            error_msg = "\n".join(f"{f}: {e}" for f, e in errors)
            raise RuntimeError(f"Failed to process files:\n{error_msg}")

        return processed_content

    def convert_to_pdf(self, output_file: Path) -> None:
        """Convert markdown files to PDF using pandoc.
        
        Args:
            output_file: Path where the output PDF will be saved
            
        Raises:
            RuntimeError: If conversion fails
            FileNotFoundError: If required dependencies are missing
        """
        try:
            self._check_dependencies()
            files = self._parse_toc()
            
            # Process files in parallel
            contents = self._process_markdown_files(files)
            
            # Create temporary combined file
            temp_md = self.output_dir / self.config.temp_filename
            self._create_combined_markdown(temp_md, contents)
            
            # Run pandoc conversion
            self._run_pandoc_conversion(temp_md, output_file)
            
        except Exception as e:
            logger.error(f"PDF conversion failed: {e}")
            raise
        finally:
            # Clean up
            self._cleanup()

    def _create_combined_markdown(self, temp_md: Path, contents: List[str]):
        """Create a combined markdown file from processed markdown content."""
        # Add title page without BOM
        title = self.toc_file.stem.replace("_", " ").title()
        combined_content = [
            f"---",
            f"title: {title}",
            f"date: {datetime.datetime.now().strftime('%Y-%m-%d')}",
            f"---",
            "\n\n",
        ]

        # Process and combine all markdown content
        for content in contents:
            lines = content.split('\n')
            processed_lines = []
            
            i = 0
            while i < len(lines):
                line = lines[i].rstrip()
                
                # Skip empty lines at the start
                if not processed_lines and not line:
                    i += 1
                    continue

                processed_lines.append(line)
                i += 1

            # Add processed content with double newlines between sections
            if processed_lines:
                combined_content.append('\n'.join(processed_lines))

        # Write combined content to temporary file with UTF-8 encoding (no BOM)
        logger.debug(f"Writing combined content with {len(combined_content)} sections")
        temp_md.write_text('\n\n'.join(combined_content), encoding='utf-8')

    def _run_pandoc_conversion(self, temp_md: Path, output_file: Path):
        """Run pandoc conversion using the combined markdown file."""
        # Get title from toc file
        title = self.toc_file.stem.replace("_", " ").title()
        
        # Create a temporary header file with proven LaTeX settings
        header_file = self.output_dir / "header.tex"
        try:
            # Simplified LaTeX header with essential settings
            header_content = r"""
\usepackage{xcolor}
\usepackage{listings}
\usepackage{geometry}
\usepackage{fancyhdr}
\usepackage{titlesec}

% Page setup
\geometry{a4paper, margin=1in}
\pagestyle{fancy}
\fancyhf{}
\fancyhead[L]{\nouppercase{\leftmark}}
\fancyhead[R]{\thepage}
\renewcommand{\headrulewidth}{0.4pt}

% Chapter and section formatting - removed numbering
\titleformat{\chapter}
{\normalfont\huge\bfseries}{}{0pt}{\Huge}
\titlespacing*{\chapter}{0pt}{-20pt}{40pt}

% Remove section numbering
\setcounter{secnumdepth}{0}

% Rest of the header content remains the same...
"""
            header_file.write_text(header_content, encoding='utf-8')
            
            # Build pandoc command - removed --number-sections
            cmd = [
                "pandoc",
                str(temp_md),
                "--pdf-engine=xelatex",
                "--toc",
                "--toc-depth=3",
                "--top-level-division=chapter",
                "-V",
                "documentclass=report",
                "-V",
                f"title={title}",
                "--highlight-style=pygments",
                "-f",
                "markdown+smart+fenced_code_blocks+auto_identifiers",
                "--listings",
                f"--include-in-header={header_file}",
                "--wrap=none",
                "-V",
                "papersize=a4",
                "-V",
                "fontsize=11pt",
                "-V",
                "geometry:margin=1in",
                "-V",
                "linkcolor=blue",
                "--variable",
                "urlcolor=blue",
                "--variable",
                "toccolor=black",
                "-V",
                "colorlinks=true",
                "--pdf-engine-opt=-shell-escape",
                "--verbose",
                "-o",
                str(output_file),
            ]

            logger.info(f"Running pandoc command: {' '.join(cmd)}")

            try:
                result = subprocess.run(
                    cmd, 
                    capture_output=True, 
                    text=True, 
                    check=True, 
                    encoding="utf-8"
                )
                
                if result.stderr:
                    logger.warning(f"Pandoc warnings: {result.stderr}")
                    
                logger.info(f"Successfully created PDF: {output_file}")
                
            except subprocess.CalledProcessError as e:
                logger.error(f"Pandoc conversion failed with return code {e.returncode}")
                logger.error(f"Error output:\n{e.stderr}")
                raise

        except Exception as e:
            logger.error(f"PDF conversion failed: {str(e)}")
            raise
        finally:
            # Clean up temporary header file
            if header_file.exists():
                header_file.unlink()

    def _cleanup(self):
        """Clean up temporary files and resources."""
        # Clean up temporary file
        if (self.output_dir / self.config.temp_filename).exists():
            (self.output_dir / self.config.temp_filename).unlink()
            logger.debug("Cleaned up temporary markdown file")


class ConverterGUI:
    """GUI interface for the markdown to PDF converter."""

    def __init__(self):
        """Initialize the GUI."""
        self.root = tk.Tk()
        self.root.title("Markdown to PDF Converter")
        self.root.geometry("800x600")

        # Initialize paths
        self.toc_path: Optional[Path] = None
        self.input_dir: Optional[Path] = None
        self.output_file: Optional[Path] = None
        self.css_file: Optional[Path] = None

        self._create_widgets()

    def _create_widgets(self):
        """Create and arrange GUI widgets."""
        # Main container
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # File Selection Frame
        file_frame = ttk.LabelFrame(main_frame, text="File Selection", padding="10")
        file_frame.pack(fill=tk.X, pady=(0, 10))

        # TOC File Selection
        toc_frame = ttk.Frame(file_frame)
        toc_frame.pack(fill=tk.X, pady=5)

        ttk.Label(toc_frame, text="Table of Contents:").pack(side=tk.LEFT)
        self.toc_var = tk.StringVar()
        ttk.Entry(toc_frame, textvariable=self.toc_var, width=50).pack(
            side=tk.LEFT, padx=5
        )
        ttk.Button(toc_frame, text="Browse", command=self._select_toc).pack(
            side=tk.LEFT
        )

        # Input Directory Selection
        input_frame = ttk.Frame(file_frame)
        input_frame.pack(fill=tk.X, pady=5)

        ttk.Label(input_frame, text="Input Directory:").pack(side=tk.LEFT)
        self.input_var = tk.StringVar()
        ttk.Entry(input_frame, textvariable=self.input_var, width=50).pack(
            side=tk.LEFT, padx=5
        )
        ttk.Button(input_frame, text="Browse", command=self._select_input_dir).pack(
            side=tk.LEFT
        )

        # Output File Selection
        output_frame = ttk.Frame(file_frame)
        output_frame.pack(fill=tk.X, pady=5)

        ttk.Label(output_frame, text="Output PDF:").pack(side=tk.LEFT)
        self.output_var = tk.StringVar()
        ttk.Entry(output_frame, textvariable=self.output_var, width=50).pack(
            side=tk.LEFT, padx=5
        )
        ttk.Button(output_frame, text="Browse", command=self._select_output).pack(
            side=tk.LEFT
        )

        # CSS File Selection (Optional)
        css_frame = ttk.Frame(file_frame)
        css_frame.pack(fill=tk.X, pady=5)

        ttk.Label(css_frame, text="Custom CSS (Optional):").pack(side=tk.LEFT)
        self.css_var = tk.StringVar()
        ttk.Entry(css_frame, textvariable=self.css_var, width=50).pack(
            side=tk.LEFT, padx=5
        )
        ttk.Button(css_frame, text="Browse", command=self._select_css).pack(
            side=tk.LEFT
        )

        # Log Display
        log_frame = ttk.LabelFrame(main_frame, text="Logs", padding="10")
        log_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))

        self.log_display = scrolledtext.ScrolledText(log_frame, height=15)
        self.log_display.pack(fill=tk.BOTH, expand=True)

        # Convert Button
        self.convert_button = ttk.Button(
            main_frame,
            text="Convert to PDF",
            command=self._start_conversion,
            state=tk.DISABLED,
        )
        self.convert_button.pack(pady=10)

    def _select_toc(self):
        """Handle table of contents file selection."""
        file_path = filedialog.askopenfilename(
            title="Select Table of Contents",
            filetypes=[("Markdown files", "*.md"), ("All files", "*.*")],
        )
        if file_path:
            self.toc_path = Path(file_path)
            self.toc_var.set(str(self.toc_path))
            self._validate_inputs()
            self.update_log(f"Selected table of contents: {self.toc_path}")

    def _select_input_dir(self):
        """Handle input directory selection."""
        dir_path = filedialog.askdirectory(title="Select Input Directory")
        if dir_path:
            self.input_dir = Path(dir_path)
            self.input_var.set(str(self.input_dir))
            self._validate_inputs()
            self.update_log(f"Selected input directory: {self.input_dir}")

    def _select_output(self):
        """Handle output PDF file selection."""
        file_path = filedialog.asksaveasfilename(
            title="Save PDF As",
            defaultextension=".pdf",
            filetypes=[("PDF files", "*.pdf"), ("All files", "*.*")],
        )
        if file_path:
            self.output_file = Path(file_path)
            self.output_var.set(str(self.output_file))
            self._validate_inputs()
            self.update_log(f"Selected output file: {self.output_file}")

    def _select_css(self):
        """Handle CSS file selection."""
        file_path = filedialog.askopenfilename(
            title="Select CSS File",
            filetypes=[("CSS files", "*.css"), ("All files", "*.*")],
        )
        if file_path:
            self.css_file = Path(file_path)
            self.css_var.set(str(self.css_file))
            self.update_log(f"Selected CSS file: {self.css_file}")

    def _validate_inputs(self):
        """Validate inputs and enable/disable convert button."""
        valid = (
            self.toc_path is not None
            and self.toc_path.exists()
            and self.input_dir is not None
            and self.input_dir.exists()
            and self.output_file is not None
        )
        self.convert_button.config(state=tk.NORMAL if valid else tk.DISABLED)

    def _start_conversion(self):
        """Start the PDF conversion process."""
        self.convert_button.config(state=tk.DISABLED)
        self.update_log("Starting conversion...")

        def conversion_thread():
            try:
                converter = MarkdownToPDFConverter(
                    self.toc_path, self.input_dir, self.css_file
                )
                converter.convert_to_pdf(self.output_file)
                self.update_log("Conversion completed successfully!")

            except Exception as e:
                self.update_log(f"Error during conversion: {str(e)}")
                logger.error(f"Conversion error: {e}")

            finally:
                self.root.after(0, lambda: self.convert_button.config(state=tk.NORMAL))

        # Run conversion in separate thread
        thread = threading.Thread(target=conversion_thread)
        thread.daemon = True
        thread.start()

    def update_log(self, message: str):
        """Update the log display."""
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        formatted_message = f"[{timestamp}] {message}"

        def update():
            self.log_display.insert(tk.END, f"{formatted_message}\n")
            self.log_display.see(tk.END)

        self.root.after(0, update)
        logger.info(message)

    def run(self):
        """Start the GUI main loop."""
        self.root.mainloop()


def main():
    """Main entry point."""
    try:
        app = ConverterGUI()
        app.run()
    except Exception as e:
        logger.error(f"Application error: {e}")
        raise


if __name__ == "__main__":
    main()
