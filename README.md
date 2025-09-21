# ThemeClouds

A Python package for extracting and analyzing themes from documents using Large Language Models (LLMs). This package processes text documents, identifies key themes using configurable LLM prompts, and creates visualizations of theme frequency.

## Features

- **LLM-powered theme extraction**: Uses OpenAI GPT models (with extensible interface for other LLMs)
- **Configurable prompts**: Customize extraction and matching prompts via JSON configuration
- **Theme frequency analysis**: Count how often themes appear across document collections
- **Rich visualizations**: Generate word clouds and bar charts of theme frequencies
- **Extensible architecture**: Easy to add support for new LLM providers
- **Multiple input formats**: Support for text files, JSON, and direct string input

## Installation

Install from source:

```bash
git clone <repository-url>
cd theme-extractor
pip install -e .
```

## Quick Start

```python
from themeclouds import ThemeExtractor
import os

# Download GGUF of your LLM of choice
llm_path = "/your/llm/path"

# Sample documents
documents = [
    "Your first document text here...",
    "Your second document text here...",
    # ... more documents
]

# Initialize and process
extractor = ThemeExtractor(llm_path=llm_path)
theme_counts = extractor.process_documents(documents)

# Generate visualizations
extractor.generate_visualizations(output_dir="results")

# Save results
extractor.save_results("results/analysis.json")
```

## Configuration

### Custom Prompts

Create a `prompt_config.json` file to customize the prompts:

```python
from themeclouds.utils import create_sample_config

# Generate sample configuration file
create_sample_config("my_prompts.json")

# Use custom configuration
extractor = ThemeExtractor(prompt_config_path="my_prompts.json")
```

### Custom LLM Settings

```python
from themeclouds import ThemeExtractor
from themeclouds.llm_interface import OpenAIInterface

# Custom OpenAI settings
llm = OpenAIInterface(
    model="gpt-4",
    max_retries=5,
    delay_between_calls=2.0
)

extractor = ThemeExtractor(llm_interface=llm)
```

### Visualization Settings

```python
extractor = ThemeExtractor(
    wordcloud_config={
        "width": 1200,
        "height": 600,
        "background_color": "black",
        "colormap": "plasma"
    }
)
```

## Package Structure

```
themeclouds/
├── __init__.py              # Package initialization
├── core.py                  # Main ThemeExtractor class
├── llm_interface.py         # LLM communication interfaces
├── theme_analyzer.py        # Theme extraction and analysis logic
├── visualizer.py           # Visualization generation
├── prompt_config.py        # Prompt configuration management
└── utils.py                # Utility functions
```

## Advanced Usage

### Processing Large Document Collections

```python
from themeclouds.utils import load_documents_from_file, preprocess_text

# Load documents from file
documents = load_documents_from_file("my_documents.txt")

# Preprocess for optimal LLM processing
processed_docs = [preprocess_text(doc) for doc in documents]

# Process with custom settings
extractor = ThemeExtractor()
results = extractor.process_documents(processed_docs)
```

### Custom LLM Interface

Extend the `LLMInterface` class to add support for other LLM providers:

```python
from themeclouds.llm_interface import LLMInterface

class CustomLLMInterface(LLMInterface):
    def generate_response(self, system_prompt: str, user_prompt: str) -> str:
        # Implement your custom LLM API call here
        return "Your LLM response"

# Use your custom interface
extractor = ThemeExtractor(llm_interface=CustomLLMInterface())
```

### Analyzing Results

```python
# Get detailed results
results = extractor.get_results()

print(f"Total themes found: {results['summary']['total_themes']}")
print(f"Most common themes: {results['summary']['most_common_themes']}")

# Access per-document themes
for doc_id, themes in results['document_themes'].items():
    print(f"Document {doc_id}: {themes}")
```

## API Reference

### ThemeExtractor

Main class that orchestrates the entire theme extraction workflow.

**Methods:**
- `process_documents(documents: List[str]) -> Dict[str, int]`: Process documents and return theme counts
- `generate_visualizations(...)`: Create word clouds and bar charts
- `get_results() -> Dict`: Get complete analysis results
- `save_results(output_path: str)`: Save results to JSON file

### PromptConfig

Manages LLM prompt configurations.

**Methods:**
- `load_config(config_path: str)`: Load custom prompts from JSON
- `get_theme_extraction_prompts() -> Dict[str, str]`: Get extraction prompts
- `get_theme_matching_prompts() -> Dict[str, str]`: Get matching prompts

### WordCloudGenerator

Creates visualizations from theme frequency data.

**Methods:**
- `generate_wordcloud(theme_counts, ...)`: Create word cloud visualization
- `generate_bar_chart(theme_counts, ...)`: Create bar chart visualization

## Requirements

- Python 3.8+
- OpenAI API key (for default LLM interface)
- Required packages: `openai`, `wordcloud`, `matplotlib`

## Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for bugs and feature requests.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
