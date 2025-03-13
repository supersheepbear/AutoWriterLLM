"""GUI module for AutoWriterLLM.

This module provides a graphical user interface for the AutoWriterLLM content generator.
"""

import logging
import datetime
import threading
import tkinter as tk
from tkinter import ttk, scrolledtext, filedialog, messagebox
from pathlib import Path
import yaml
import time
import re

from autowriterllm.ai_content_generator import ContentGenerator
from autowriterllm.exceptions import ContentGenerationError

logger = logging.getLogger(__name__)

class ContentGeneratorGUI:
    """GUI for the content generator.
    
    This class provides a graphical user interface for the content generator,
    allowing users to select files, configure generation options, and monitor
    progress.
    
    Attributes:
        root (tk.Tk): The root Tkinter window
        topic_var (tk.StringVar): Variable for the topic input
        level_var (tk.StringVar): Variable for the level selection
        language_var (tk.StringVar): Variable for the language selection
        toc_path (Optional[Path]): Path to the table of contents file
        summary_path (Optional[Path]): Path to the summary file
        output_path (Optional[Path]): Path to the output directory
        generator (Optional[ContentGenerator]): The content generator instance
    """
    
    def __init__(self):
        """Initialize the GUI."""
        self.root = tk.Tk()
        self.root.title("AutoWriterLLM - Content Generator")
        self.root.geometry("800x600")
        
        # Initialize variables
        self.topic_var = tk.StringVar()
        self.level_var = tk.StringVar(value="intermediate")
        self.output_path_var = tk.StringVar()
        self.progress_var = tk.DoubleVar()
        self.language_var = tk.StringVar(value="auto")
        self.toc_path = None
        self.summary_path = None
        self.output_path = None
        self.generator = None
        self.provider_var = tk.StringVar()
        self.model_var = tk.StringVar()
        
        # Create the main frame
        main_frame = ttk.Frame(self.root, padding=10)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create the notebook for tabs
        notebook = ttk.Notebook(main_frame)
        notebook.pack(fill=tk.BOTH, expand=True)
        
        # Create tabs
        plan_tab = ttk.Frame(notebook)
        generate_tab = ttk.Frame(notebook)
        settings_tab = ttk.Frame(notebook)
        
        notebook.add(plan_tab, text="Plan Book")
        notebook.add(generate_tab, text="Generate Content")
        notebook.add(settings_tab, text="Settings")
        
        # Plan tab
        self._setup_plan_tab(plan_tab)
        
        # Generate tab
        self._setup_generate_tab(generate_tab)
        
        # Settings tab
        self._setup_settings_tab(settings_tab)
        
        # Log area
        log_frame = ttk.LabelFrame(main_frame, text="Log")
        log_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        self.log_text = scrolledtext.ScrolledText(log_frame, height=10)
        self.log_text.pack(fill=tk.BOTH, expand=True)
        
        # Load configuration
        self._load_config()
        
        # Update log with startup message
        self.update_log("Application started")
    
    def _setup_plan_tab(self, parent):
        """Set up the plan tab UI.
        
        Args:
            parent: The parent widget
        """
        # Topic input
        topic_frame = ttk.Frame(parent)
        topic_frame.pack(fill=tk.X, pady=10)
        
        ttk.Label(topic_frame, text="Topic:").pack(side=tk.LEFT, padx=5)
        topic_entry = ttk.Entry(topic_frame, textvariable=self.topic_var, width=50)
        topic_entry.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        
        # Level selection
        level_frame = ttk.Frame(parent)
        level_frame.pack(fill=tk.X, pady=10)
        
        ttk.Label(level_frame, text="Level:").pack(side=tk.LEFT, padx=5)
        levels = ["beginner", "intermediate", "advanced"]
        level_combo = ttk.Combobox(level_frame, textvariable=self.level_var, values=levels)
        level_combo.pack(side=tk.LEFT, padx=5)
        
        # Language selection
        language_frame = ttk.Frame(parent)
        language_frame.pack(fill=tk.X, pady=10)
        
        ttk.Label(language_frame, text="Language:").pack(side=tk.LEFT, padx=5)
        languages = ["auto", "English", "Chinese"]
        language_combo = ttk.Combobox(language_frame, textvariable=self.language_var, values=languages, state="readonly")
        language_combo.pack(side=tk.LEFT, padx=5)
        
        # Provider and model selection
        model_frame = ttk.Frame(parent)
        model_frame.pack(fill=tk.X, pady=10)
        
        ttk.Label(model_frame, text="Provider:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        self.plan_provider_combo = ttk.Combobox(model_frame, textvariable=self.provider_var, state="readonly")
        self.plan_provider_combo.grid(row=0, column=1, padx=5, pady=5, sticky=tk.EW)
        self.plan_provider_combo.bind("<<ComboboxSelected>>", self._update_model_list)
        
        ttk.Label(model_frame, text="Model:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
        self.plan_model_combo = ttk.Combobox(model_frame, textvariable=self.model_var, state="readonly")
        self.plan_model_combo.grid(row=1, column=1, padx=5, pady=5, sticky=tk.EW)
        
        model_frame.columnconfigure(1, weight=1)
        
        # Output directory
        output_frame = ttk.Frame(parent)
        output_frame.pack(fill=tk.X, pady=10)
        
        ttk.Label(output_frame, text="Output Directory:").pack(side=tk.LEFT, padx=5)
        output_entry = ttk.Entry(output_frame, textvariable=self.output_path_var, width=40)
        output_entry.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        ttk.Button(output_frame, text="Browse...", command=self._select_output_dir).pack(side=tk.LEFT, padx=5)
        
        # Generate button
        button_frame = ttk.Frame(parent)
        button_frame.pack(fill=tk.X, pady=20)
        
        ttk.Button(button_frame, text="Generate Book Plan", command=self._generate_book_plan).pack(pady=10)
    
    def _setup_generate_tab(self, parent):
        """Set up the generate tab UI.
        
        Args:
            parent: The parent widget
        """
        # File selection
        file_frame = ttk.Frame(parent)
        file_frame.pack(fill=tk.X, pady=10)
        
        ttk.Label(file_frame, text="Table of Contents:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        self.toc_entry = ttk.Entry(file_frame, width=40)
        self.toc_entry.grid(row=0, column=1, padx=5, pady=5, sticky=tk.EW)
        ttk.Button(file_frame, text="Browse...", command=self._select_toc_file).grid(row=0, column=2, padx=5, pady=5)
        
        ttk.Label(file_frame, text="Summary File:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
        self.summary_entry = ttk.Entry(file_frame, width=40)
        self.summary_entry.grid(row=1, column=1, padx=5, pady=5, sticky=tk.EW)
        ttk.Button(file_frame, text="Browse...", command=self._select_summary_file).grid(row=1, column=2, padx=5, pady=5)
        
        ttk.Label(file_frame, text="Output Directory:").grid(row=2, column=0, sticky=tk.W, padx=5, pady=5)
        self.output_entry = ttk.Entry(file_frame, width=40)
        self.output_entry.grid(row=2, column=1, padx=5, pady=5, sticky=tk.EW)
        ttk.Button(file_frame, text="Browse...", command=self._select_output_dir).grid(row=2, column=2, padx=5, pady=5)
        
        file_frame.columnconfigure(1, weight=1)
        
        # Provider and model selection
        model_frame = ttk.Frame(parent)
        model_frame.pack(fill=tk.X, pady=10)
        
        ttk.Label(model_frame, text="Provider:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        self.provider_combo = ttk.Combobox(model_frame, textvariable=self.provider_var, state="readonly")
        self.provider_combo.grid(row=0, column=1, padx=5, pady=5, sticky=tk.EW)
        self.provider_combo.bind("<<ComboboxSelected>>", self._update_model_list)
        
        ttk.Label(model_frame, text="Model:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
        self.model_combo = ttk.Combobox(model_frame, textvariable=self.model_var, state="readonly")
        self.model_combo.grid(row=1, column=1, padx=5, pady=5, sticky=tk.EW)
        
        ttk.Label(model_frame, text="Language:").grid(row=2, column=0, sticky=tk.W, padx=5, pady=5)
        self.generate_language_combo = ttk.Combobox(model_frame, textvariable=self.language_var, state="readonly")
        self.generate_language_combo.grid(row=2, column=1, padx=5, pady=5, sticky=tk.EW)
        self.generate_language_combo["values"] = ["auto", "English", "Chinese"]
        
        model_frame.columnconfigure(1, weight=1)
        
        # Progress bar
        progress_frame = ttk.Frame(parent)
        progress_frame.pack(fill=tk.X, pady=10)
        
        self.progress_bar = ttk.Progressbar(progress_frame, variable=self.progress_var, maximum=100)
        self.progress_bar.pack(fill=tk.X, padx=5, pady=5)
        
        # Start button
        button_frame = ttk.Frame(parent)
        button_frame.pack(fill=tk.X, pady=10)
        
        self.start_button = ttk.Button(button_frame, text="Start Generation", command=self.start_generation, state=tk.DISABLED)
        self.start_button.pack(pady=10)
    
    def _setup_settings_tab(self, parent):
        """Set up the settings tab UI.
        
        Args:
            parent: The parent widget
        """
        # Multi-part generation
        multi_part_frame = ttk.Frame(parent)
        multi_part_frame.pack(fill=tk.X, pady=10)
        
        self.multi_part_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(multi_part_frame, text="Enable multi-part generation", variable=self.multi_part_var).pack(anchor=tk.W, padx=5)
        
        # Parts per section
        parts_frame = ttk.Frame(parent)
        parts_frame.pack(fill=tk.X, pady=10)
        
        ttk.Label(parts_frame, text="Parts per section:").pack(side=tk.LEFT, padx=5)
        self.parts_var = tk.IntVar(value=5)
        parts_spinbox = ttk.Spinbox(parts_frame, from_=1, to=10, textvariable=self.parts_var, width=5)
        parts_spinbox.pack(side=tk.LEFT, padx=5)
        
        # Max tokens
        tokens_frame = ttk.Frame(parent)
        tokens_frame.pack(fill=tk.X, pady=10)
        
        ttk.Label(tokens_frame, text="Max tokens per part:").pack(side=tk.LEFT, padx=5)
        self.tokens_var = tk.IntVar(value=8192)
        tokens_spinbox = ttk.Spinbox(tokens_frame, from_=1000, to=16000, increment=1000, textvariable=self.tokens_var, width=7)
        tokens_spinbox.pack(side=tk.LEFT, padx=5)
        
        # Save button
        button_frame = ttk.Frame(parent)
        button_frame.pack(fill=tk.X, pady=20)
        
        ttk.Button(button_frame, text="Save Settings", command=self._save_settings).pack(pady=10)
    
    def _load_config(self):
        """Load configuration from config file."""
        try:
            config_path = Path("config.yaml")
            if config_path.exists():
                with open(config_path) as f:
                    config = yaml.safe_load(f)
                
                # Load provider and model options
                providers = config.get("providers", {})
                provider_list = list(providers.keys())
                
                # Set values for both provider combos
                self.provider_combo["values"] = provider_list
                self.plan_provider_combo["values"] = provider_list
                
                # Set default provider
                default_provider = config.get("provider")
                if default_provider in providers:
                    self.provider_var.set(default_provider)
                    self._update_model_list()
                
                # Load other settings
                self.multi_part_var.set(config.get("multi_part_generation", True))
                self.parts_var.set(config.get("parts_per_section", 5))
                self.tokens_var.set(config.get("max_tokens_per_part", 8192))
                
                self.update_log("Configuration loaded successfully")
            else:
                self.update_log("No configuration file found, using defaults")
        except Exception as e:
            self.update_log(f"Error loading configuration: {e}")
    
    def _save_settings(self):
        """Save settings to config file."""
        try:
            config_path = Path("config.yaml")
            
            # Load existing config if available
            if config_path.exists():
                with open(config_path) as f:
                    config = yaml.safe_load(f)
            else:
                config = {}
            
            # Update settings
            config["multi_part_generation"] = self.multi_part_var.get()
            config["parts_per_section"] = self.parts_var.get()
            config["max_tokens_per_part"] = self.tokens_var.get()
            
            # Save provider and model if selected
            if self.provider_var.get():
                config["provider"] = self.provider_var.get()
            
            # Write config
            with open(config_path, "w") as f:
                yaml.dump(config, f)
            
            self.update_log("Settings saved successfully")
            messagebox.showinfo("Settings", "Settings saved successfully")
        except Exception as e:
            self.update_log(f"Error saving settings: {e}")
            messagebox.showerror("Error", f"Failed to save settings: {e}")
    
    def _update_model_list(self, event=None):
        """Update the model list based on selected provider."""
        try:
            provider = self.provider_var.get()
            if not provider:
                return
                
            # Load config to get provider models
            with open("config.yaml") as f:
                config = yaml.safe_load(f)
            
            providers = config.get("providers", {})
            provider_config = providers.get(provider, {})
            models = provider_config.get("models", [])
            
            # Get list of model names
            model_names = [model["name"] for model in models]
            
            # Update both model combos
            self.model_combo["values"] = model_names
            self.plan_model_combo["values"] = model_names
            
            # Set first model as default if available
            if model_names:
                self.model_var.set(model_names[0])
                
        except Exception as e:
            self.update_log(f"Error updating model list: {e}")
            messagebox.showerror("Error", f"Failed to update model list: {e}")
    
    def _select_toc_file(self):
        """Open file dialog to select a table of contents file."""
        try:
            file_path = filedialog.askopenfilename(
                title="Select Table of Contents File",
                filetypes=[("Markdown Files", "*.md"), ("All Files", "*.*")]
            )
            
            if file_path:
                self.toc_path = Path(file_path)
                self.toc_entry.delete(0, tk.END)
                self.toc_entry.insert(0, str(self.toc_path))
                self.update_log(f"Selected TOC file: {self.toc_path}")
                
                # Try to auto-detect summary file
                self._auto_detect_summary()
                
                # Validate inputs
                self._validate_inputs()
        except Exception as e:
            self.update_log(f"Error selecting TOC file: {e}")
    
    def _select_output_dir(self):
        """Open file dialog to select an output directory."""
        try:
            dir_path = filedialog.askdirectory(title="Select Output Directory")
            
            if dir_path:
                self.output_path = Path(dir_path)
                
                # Update both output entries
                self.output_path_var.set(str(self.output_path))
                self.output_entry.delete(0, tk.END)
                self.output_entry.insert(0, str(self.output_path))
                
                self.update_log(f"Selected output directory: {self.output_path}")
                
                # Validate inputs
                self._validate_inputs()
        except Exception as e:
            self.update_log(f"Error selecting output directory: {e}")
    
    def _select_summary_file(self):
        """Open file dialog to select a summary file."""
        try:
            file_path = filedialog.askopenfilename(
                title="Select Summary File",
                filetypes=[("Markdown Files", "*.md"), ("All Files", "*.*")]
            )
            
            if file_path:
                self.summary_path = Path(file_path)
                self.summary_entry.delete(0, tk.END)
                self.summary_entry.insert(0, str(self.summary_path))
                self.update_log(f"Selected summary file: {self.summary_path}")
                
                # Validate inputs
                self._validate_inputs()
        except Exception as e:
            self.update_log(f"Error selecting summary file: {e}")
    
    def _auto_detect_summary(self):
        """Try to auto-detect the summary file based on the TOC file."""
        try:
            if not self.toc_path:
                return
            
            # Construct expected summary file path
            toc_stem = self.toc_path.stem
            if "table_of_contents" in toc_stem or "toc" in toc_stem.lower():
                summary_path = self.toc_path.parent / "book_summary.md"
                
                if summary_path.exists():
                    self.summary_path = summary_path
                    self.summary_entry.delete(0, tk.END)
                    self.summary_entry.insert(0, str(self.summary_path))
                    self.update_log(f"Auto-detected summary file: {self.summary_path}")
                else:
                    self.update_log("Could not auto-detect summary file")
                    messagebox.showwarning("Summary File", "Could not auto-detect summary file. Please select manually.")
        except Exception as e:
            self.update_log(f"Error auto-detecting summary file: {e}")
    
    def _validate_inputs(self):
        """Check if all required inputs are valid and enable/disable the start button."""
        if self.toc_path and self.output_path:
            # Summary is optional
            self.start_button["state"] = tk.NORMAL
        else:
            self.start_button["state"] = tk.DISABLED
    
    def update_log(self, message: str):
        """Update the log display with a timestamped message.
        
        Args:
            message: The message to add to the log
        """
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        log_message = f"[{timestamp}] {message}\n"
        
        self.log_text.configure(state=tk.NORMAL)
        self.log_text.insert(tk.END, log_message)
        self.log_text.see(tk.END)
        self.log_text.configure(state=tk.DISABLED)
        
        # Also log to the logger
        logger.info(message)
    
    def start_generation(self):
        """Start the content generation process."""
        try:
            # Check if we have the required files
            if not self.toc_path or not self.toc_path.exists():
                messagebox.showerror("Error", "Table of contents file not found")
                return
            
            if not self.output_path:
                messagebox.showerror("Error", "Output directory not specified")
                return
            
            # Disable inputs during generation
            self._disable_inputs()
            
            # Start generation in a separate thread
            threading.Thread(target=self._generation_process, daemon=True).start()
            
        except Exception as e:
            self.update_log(f"Error starting generation: {e}")
            messagebox.showerror("Error", f"Failed to start generation: {e}")
            self._enable_inputs()
    
    def _generation_process(self):
        """Handle the content generation process in a separate thread."""
        try:
            self.update_log("Starting content generation process")
            
            # Initialize the generator
            self.generator = ContentGenerator(self.toc_path, self.output_path)
            
            # Set the selected model if specified
            if self.provider_var.get() and self.model_var.get():
                self.generator.set_model(self.provider_var.get(), self.model_var.get())
                self.update_log(f"Using model: {self.provider_var.get()}/{self.model_var.get()}")
            
            # Parse the TOC
            self.update_log("Parsing table of contents")
            self.generator.parse_toc()
            
            # Get language setting
            language = self.language_var.get() if hasattr(self, 'language_var') else "auto"
            self.update_log(f"Using language: {language if language != 'auto' else 'auto-detected'}")
            
            # Generate content with progress updates
            self.update_log("Starting content generation")
            self.generator.generate_all_content(
                progress_callback=self.update_progress,
                language=language
            )
            
            # Complete
            self.update_log("Content generation completed successfully")
            messagebox.showinfo("Success", "Content generation completed successfully")
            
        except Exception as e:
            self.update_log(f"Error during generation: {e}")
            messagebox.showerror("Error", f"Generation failed: {e}")
            
        finally:
            # Re-enable inputs
            self._enable_inputs()
    
    def update_progress(self, current: int, total: int, progress: float = None):
        """Update the progress bar and log with the current progress.
        
        Args:
            current: The current unit number
            total: The total number of units
            progress: Optional pre-calculated progress value (0-1)
        """
        if total > 0:
            if progress is not None:
                # Use provided progress value
                progress_percent = progress * 100
            else:
                # Calculate progress if not provided
                progress_percent = (current / total) * 100
            self.progress_var.set(progress_percent)
            if current % 5 == 0 or current == total:  # Log every 5 units or at completion
                self.update_log(f"Generation progress: {current}/{total} units ({progress_percent:.1f}%)")
        else:
            self.progress_var.set(0)
            self.update_log("No units to generate")
    
    def _disable_inputs(self):
        """Disable input controls during generation."""
        self.start_button["state"] = tk.DISABLED
        self.provider_combo["state"] = tk.DISABLED
        self.model_combo["state"] = tk.DISABLED
        self.generate_language_combo["state"] = tk.DISABLED
        self.plan_provider_combo["state"] = tk.DISABLED
        self.plan_model_combo["state"] = tk.DISABLED
    
    def _enable_inputs(self):
        """Enable input controls after generation."""
        self.start_button["state"] = tk.NORMAL
        self.provider_combo["state"] = "readonly"
        self.model_combo["state"] = "readonly"
        self.generate_language_combo["state"] = "readonly"
        self.plan_provider_combo["state"] = "readonly"
        self.plan_model_combo["state"] = "readonly"
    
    def _generate_book_plan(self):
        """Generate a book plan based on user input."""
        try:
            # Validate topic
            topic = self.topic_var.get().strip()
            if not self._validate_topic_input(topic):
                return
            
            # Get level
            level = self.level_var.get()
            
            # Get language
            language = self.language_var.get()
            
            # Get provider and model
            provider = self.provider_var.get()
            model = self.model_var.get()
            
            if not provider:
                messagebox.showerror("Error", "Please select an AI provider")
                return
                
            if not model:
                messagebox.showerror("Error", "Please select a model")
                return
            
            # Get output directory
            output_dir = self.output_path_var.get()
            if not output_dir:
                messagebox.showerror("Error", "Please select an output directory")
                return
            
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            self.update_log(f"Generating book plan for '{topic}' at {level} level in {language if language != 'auto' else 'auto-detected language'} using {provider}/{model}")
            
            # Load config first
            config_path = Path("config.yaml")
            if not config_path.exists():
                messagebox.showerror("Error", "Configuration file not found")
                return
                
            with open(config_path) as f:
                config = yaml.safe_load(f)
            
            # Set the provider in the config
            config["provider"] = provider
            
            # Initialize generator with the config
            self.generator = ContentGenerator(None, output_path)
            self.generator.config = config
            
            # Set the provider and model
            self.generator.set_model(provider, model)
            
            # Generate book plan with specified provider and model
            toc_content, summary_content = self.generator.generate_book_plan(
                topic=topic,
                level=level,
                provider_name=provider,
                model_name=model,
                language=language
            )
            
            # Save files
            toc_file = output_path / "table_of_contents.md"
            summary_file = output_path / "book_summary.md"
            
            # Create backups if files exist
            if toc_file.exists():
                backup_path = toc_file.with_suffix(f".bak_{int(time.time())}")
                toc_file.rename(backup_path)
                self.update_log(f"Created backup of existing TOC: {backup_path}")
            
            if summary_file.exists():
                backup_path = summary_file.with_suffix(f".bak_{int(time.time())}")
                summary_file.rename(backup_path)
                self.update_log(f"Created backup of existing summary: {backup_path}")
            
            # Write new files
            toc_file.write_text(toc_content, encoding='utf-8')
            summary_file.write_text(summary_content, encoding='utf-8')
            
            self.update_log(f"Book plan generated successfully")
            self.update_log(f"Table of contents saved to: {toc_file}")
            self.update_log(f"Book summary saved to: {summary_file}")
            
            # Show success message
            messagebox.showinfo(
                "Success", 
                f"Book plan generated successfully!\n\nFiles saved to:\n{toc_file}\n{summary_file}"
            )
            
        except Exception as e:
            self.update_log(f"Error: Failed to generate book plan: {e}")
            messagebox.showerror("Error", f"Failed to generate book plan: {e}")
    
    def _validate_topic_input(self, topic: str) -> bool:
        """Validate topic input string.
        
        Args:
            topic (str): Topic string to validate
            
        Returns:
            bool: True if valid, False otherwise
            
        Note:
            Validates the topic string for:
            - Minimum length (3 characters)
            - Maximum length (500 characters)
            - Contains actual text characters
            - Not just special characters or whitespace
        """
        # Check if empty or too short
        if not topic or len(topic.strip()) < 3:
            messagebox.showwarning(
                "Invalid Input",
                "Topic description must be at least 3 characters long"
            )
            logger.warning("Topic validation failed: Too short")
            return False
        
        # Check for meaningful content (not just special characters)
        #if not re.search(r'[a-zA-Z]', topic):
        #    messagebox.showwarning(
        #        "Invalid Input",
        #        "Topic must contain some text characters"
        #    )
        #    logger.warning("Topic validation failed: No text characters")
        #    return False
        
        # Check maximum length
        if len(topic) > 500:
            messagebox.showwarning(
                "Invalid Input",
                "Topic description is too long (max 500 characters)"
            )
            logger.warning("Topic validation failed: Too long")
            return False
        
        logger.debug(f"Topic validation passed: {topic}")
        return True
    
    def run(self) -> None:
        """Start the GUI main loop."""
        self.root.mainloop()


def main():
    """Main entry point for the application."""
    try:
        app = ContentGeneratorGUI()
        app.run()
    except Exception as e:
        logger.error(f"Application error: {e}")
        raise


if __name__ == "__main__":
    main() 