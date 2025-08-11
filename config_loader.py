"""
Configuration loader for managing prompts and settings.
Supports loading prompts from YAML files and plain text files.
"""

import yaml
from pathlib import Path
from typing import Dict, Any, Optional, List
import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class PromptConfig:
    """Configuration for a single prompt."""
    name: str
    description: str
    template: str
    default_num_topics: int = 10
    system_message: str = "You are an expert at analyzing documents and identifying key topics."
    tags: List[str] = field(default_factory=list)


@dataclass
class AppConfig:
    """Main application configuration."""
    # LLM settings
    model_id: str = "meta-llama/Llama-3.3-70B-Instruct"
    use_quantization: bool = True
    max_new_tokens: int = 800
    temperature: float = 0.3
    
    # File paths
    input_dir: str = "data/input"
    output_dir: str = "data/output"
    
    # Processing settings
    default_num_topics: int = 10
    file_encoding: str = "utf-8"
    
    # Available prompts
    prompts: Dict[str, PromptConfig] = field(default_factory=dict)


class ConfigLoader:
    """Loads configuration from YAML and text files."""
    
    def __init__(self, config_dir: str = "config"):
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(parents=True, exist_ok=True)
    
    def load_config(self, config_file: str = "config.yaml") -> AppConfig:
        """
        Load main application configuration from YAML file.
        
        Args:
            config_file: Name of the config file in config_dir
            
        Returns:
            AppConfig object
        """
        config_path = self.config_dir / config_file
        
        if not config_path.exists():
            logger.warning(f"Config file not found: {config_path}. Using defaults.")
            return self._create_default_config()
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config_data = yaml.safe_load(f) or {}
            
            # Load prompts
            prompts = self.load_all_prompts()
            
            # Create config object
            config = AppConfig(
                model_id=config_data.get('model_id', AppConfig.model_id),
                use_quantization=config_data.get('use_quantization', AppConfig.use_quantization),
                max_new_tokens=config_data.get('max_new_tokens', AppConfig.max_new_tokens),
                temperature=config_data.get('temperature', AppConfig.temperature),
                input_dir=config_data.get('input_dir', AppConfig.input_dir),
                output_dir=config_data.get('output_dir', AppConfig.output_dir),
                default_num_topics=config_data.get('default_num_topics', AppConfig.default_num_topics),
                file_encoding=config_data.get('file_encoding', AppConfig.file_encoding),
                prompts=prompts
            )
            
            logger.info(f"Loaded configuration with {len(prompts)} prompts")
            return config
            
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            return self._create_default_config()
    
    def load_all_prompts(self) -> Dict[str, PromptConfig]:
        """Load all prompts from the prompts directory."""
        prompts_dir = self.config_dir / "prompts"
        prompts_dir.mkdir(exist_ok=True)
        
        prompts = {}
        
        # Load prompts from YAML files
        for yaml_file in prompts_dir.glob("*.yaml"):
            try:
                prompt_configs = self._load_prompts_from_yaml(yaml_file)
                prompts.update(prompt_configs)
            except Exception as e:
                logger.error(f"Failed to load prompts from {yaml_file}: {e}")
        
        # Load prompts from text files
        for txt_file in prompts_dir.glob("*.txt"):
            try:
                prompt_config = self._load_prompt_from_txt(txt_file)
                if prompt_config:
                    prompts[prompt_config.name] = prompt_config
            except Exception as e:
                logger.error(f"Failed to load prompt from {txt_file}: {e}")
        
        if not prompts:
            logger.warning("No prompts found. Creating default prompts.")
            prompts = self._create_default_prompts()
            self._save_default_prompts()
        
        return prompts
    
    def _load_prompts_from_yaml(self, yaml_file: Path) -> Dict[str, PromptConfig]:
        """Load prompts from a YAML file."""
        with open(yaml_file, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        
        prompts = {}
        if 'prompts' in data:
            for prompt_data in data['prompts']:
                config = PromptConfig(
                    name=prompt_data['name'],
                    description=prompt_data.get('description', ''),
                    template=prompt_data['template'],
                    default_num_topics=prompt_data.get('default_num_topics', 10),
                    system_message=prompt_data.get('system_message', PromptConfig.system_message),
                    tags=prompt_data.get('tags', [])
                )
                prompts[config.name] = config
        
        return prompts
    
    def _load_prompt_from_txt(self, txt_file: Path) -> Optional[PromptConfig]:
        """Load a single prompt from a text file."""
        with open(txt_file, 'r', encoding='utf-8') as f:
            content = f.read().strip()
        
        if not content:
            return None
        
        # Use filename (without extension) as prompt name
        name = txt_file.stem
        
        return PromptConfig(
            name=name,
            description=f"Prompt loaded from {txt_file.name}",
            template=content,
            default_num_topics=10
        )
    
    def _create_default_config(self) -> AppConfig:
        """Create default configuration."""
        prompts = self._create_default_prompts()
        return AppConfig(prompts=prompts)
    
    def _create_default_prompts(self) -> Dict[str, PromptConfig]:
        """Create default prompt configurations."""
        webcam_prompt = PromptConfig(
            name="webcam_experience",
            description="Extract topics from webcam experience interviews",
            template="""You are analyzing interview transcripts where participants were asked to share their experiences using different webcam setups.

Your task is to identify **exactly {num_topics} meaningful and distinctive words or short phrases** that summarize participants' real experiences.

Guidelines:
- Do NOT just pick the most frequent words.
- Select words or short phrases that are **emotionally descriptive**, **technically relevant**, or **highlight distinctive qualities** (positive or negative).
- Avoid: generic words (e.g., "thing", "camera"), filler words, or phrases repeated from questions.
- Focus only on participant speech that offers insight, reaction, or description.

Return exactly {num_topics} high-quality descriptors as a bullet list.

Content to analyze:
{content}""",
            default_num_topics=10,
            tags=["interviews", "experience", "qualitative"]
        )
        
        general_prompt = PromptConfig(
            name="general_topics",
            description="Extract general topics from any document collection",
            template="""Analyze the following documents and identify the {num_topics} most important topics or themes that appear across the content.

Focus on:
- Key concepts and ideas
- Important themes
- Significant subjects discussed
- Notable patterns or recurring elements

Return exactly {num_topics} topics as a simple bullet list.

Content to analyze:
{content}""",
            default_num_topics=10,
            tags=["general", "topics", "themes"]
        )
        
        return {
            "webcam_experience": webcam_prompt,
            "general_topics": general_prompt
        }
    
    def _save_default_prompts(self):
        """Save default prompts to files for user customization."""
        prompts_dir = self.config_dir / "prompts"
        prompts_dir.mkdir(exist_ok=True)
        
        # Save as YAML for easy editing
        default_prompts_data = {
            'prompts': [
                {
                    'name': 'webcam_experience',
                    'description': 'Extract topics from webcam experience interviews',
                    'template': self._create_default_prompts()['webcam_experience'].template,
                    'default_num_topics': 10,
                    'tags': ['interviews', 'experience', 'qualitative']
                },
                {
                    'name': 'general_topics', 
                    'description': 'Extract general topics from any document collection',
                    'template': self._create_default_prompts()['general_topics'].template,
                    'default_num_topics': 10,
                    'tags': ['general', 'topics', 'themes']
                }
            ]
        }
        
        yaml_path = prompts_dir / "default_prompts.yaml"
        with open(yaml_path, 'w', encoding='utf-8') as f:
            yaml.dump(default_prompts_data, f, default_flow_style=False, sort_keys=False)
        
        logger.info(f"Created default prompts file: {yaml_path}")
    
    def save_config_template(self, config_file: str = "config.yaml"):
        """Save a template configuration file for user customization."""
        config_path = self.config_dir / config_file
        
        if config_path.exists():
            logger.info(f"Config file already exists: {config_path}")
            return
        
        template_config = {
            'model_id': 'meta-llama/Llama-3.3-70B-Instruct',
            'use_quantization': True,
            'max_new_tokens': 800,
            'temperature': 0.3,
            'input_dir': 'data/input',
            'output_dir': 'data/output',
            'default_num_topics': 10,
            'file_encoding': 'utf-8'
        }
        
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(template_config, f, default_flow_style=False)
        
        logger.info(f"Created config template: {config_path}")
    
    def get_prompt_names(self) -> List[str]:
        """Get list of available prompt names."""
        config = self.load_config()
        return list(config.prompts.keys())
    
    def get_prompt(self, prompt_name: str) -> Optional[PromptConfig]:
        """Get a specific prompt configuration."""
        config = self.load_config()
        return config.prompts.get(prompt_name)


# Example usage
if __name__ == "__main__":
    loader = ConfigLoader("config")
    
    # Create template files if they don't exist
    loader.save_config_template()
    
    # Load configuration
    config = loader.load_config()
    
    print(f"Available prompts: {list(config.prompts.keys())}")
    
    # Get a specific prompt
    webcam_prompt = loader.get_prompt("webcam_experience")
    if webcam_prompt:
        print(f"\nPrompt: {webcam_prompt.name}")
        print(f"Description: {webcam_prompt.description}")
        print(f"Template preview: {webcam_prompt.template[:100]}...")