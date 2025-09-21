"""Configuration for LLM prompts used in theme extraction."""

import json
from pathlib import Path
from typing import Dict, Any

class PromptConfig:
    """Manages prompt configurations for theme extraction."""
    
    DEFAULT_PROMPTS = {
        "theme_extraction": {
            "system_prompt": """You are an expert at analyzing text and identifying key themes. 
Extract the most important themes from the given text. Focus on:
- Main topics and concepts
- Underlying ideas and messages
- Key subjects being discussed
- Important patterns or trends mentioned

Return themes as a simple list of clear, concise phrases.""",
            
            "user_prompt": """Please extract 3-7 key themes from the following text:

Text: {text}

Return only the themes as a numbered list, with each theme being 2-4 words long."""
        },
        
        "theme_matching": {
            "system_prompt": """You are an expert at matching text content to predefined themes. 
Your task is to determine which themes from a given list are present in a piece of text.
Be precise - only match themes that are clearly present and relevant.""",
            
            "user_prompt": """Given the following text and list of themes, identify which themes are present in the text.

Text: {text}

Available themes:
{themes}

Return only the theme names that are clearly present in the text, as a simple comma-separated list."""
        }
    }
    
    def __init__(self, config_path: str = None):
        """
        Initialize prompt configuration.
        
        Args:
            config_path: Path to custom prompt configuration file (JSON)
        """
        self.config_path = config_path
        self.prompts = self.DEFAULT_PROMPTS.copy()
        
        if config_path and Path(config_path).exists():
            self.load_config(config_path)
    
    def load_config(self, config_path: str):
        """Load custom prompt configuration from JSON file."""
        try:
            with open(config_path, 'r') as f:
                custom_config = json.load(f)
            
            # Update default prompts with custom ones
            for prompt_type, prompts in custom_config.items():
                if prompt_type in self.prompts:
                    self.prompts[prompt_type].update(prompts)
                else:
                    self.prompts[prompt_type] = prompts
                    
        except Exception as e:
            print(f"Warning: Could not load config from {config_path}: {e}")
            print("Using default prompts.")
    
    def get_prompt(self, prompt_type: str, prompt_key: str) -> str:
        """Get a specific prompt."""
        return self.prompts.get(prompt_type, {}).get(prompt_key, "")
    
    def get_theme_extraction_prompts(self) -> Dict[str, str]:
        """Get prompts for theme extraction."""
        return self.prompts["theme_extraction"]
    
    def get_theme_matching_prompts(self) -> Dict[str, str]:
        """Get prompts for theme matching."""
        return self.prompts["theme_matching"]
    
    def save_config(self, output_path: str):
        """Save current configuration to JSON file."""
        with open(output_path, 'w') as f:
            json.dump(self.prompts, f, indent=2)