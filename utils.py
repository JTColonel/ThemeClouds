"""Utility functions for the theme extractor package."""

import json
from typing import List, Dict, Any
from pathlib import Path

def load_documents_from_file(file_path: str) -> List[str]:
    """
    Load documents from various file formats.
    
    Args:
        file_path: Path to the file
        
    Returns:
        List of document strings
    """
    file_path = Path(file_path)
    
    if file_path.suffix.lower() == '.txt':
        with open(file_path, 'r', encoding='utf-8') as f:
            # Assume each line is a document, or split by double newlines
            content = f.read()
            if '\n\n' in content:
                return [doc.strip() for doc in content.split('\n\n') if doc.strip()]
            else:
                return [line.strip() for line in content.split('\n') if line.strip()]
    
    elif file_path.suffix.lower() == '.json':
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            if isinstance(data, list):
                return [str(item) for item in data]
            elif isinstance(data, dict):
                # Assume values are the documents
                return [str(value) for value in data.values()]
    
    else:
        raise ValueError(f"Unsupported file format: {file_path.suffix}")

def create_sample_config(output_path: str = "prompt_config.json") -> None:
    """Create a sample prompt configuration file."""
    from .prompt_config import PromptConfig
    config = PromptConfig()
    config.save_config(output_path)
    print(f"Sample configuration saved to {output_path}")

def preprocess_text(text: str, max_length: int = 4000) -> str:
    """
    Preprocess text for LLM processing.
    
    Args:
        text: Input text
        max_length: Maximum length to truncate to
        
    Returns:
        Preprocessed text
    """
    # Basic cleaning
    text = text.strip()
    text = ' '.join(text.split())  # Normalize whitespace
    
    # Truncate if too long
    if len(text) > max_length:
        text = text[:max_length] + "..."
    
    return text