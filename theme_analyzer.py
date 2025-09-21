"""Core theme analysis functionality."""

from typing import List, Dict, Set
from collections import Counter
import re
from .llm_interface import LLMInterface
from .prompt_config import PromptConfig

class ThemeAnalyzer:
    """Analyzes themes in text documents."""
    
    def __init__(self, llm_interface: LLMInterface, prompt_config: PromptConfig = None):
        """
        Initialize theme analyzer.
        
        Args:
            llm_interface: LLM interface to use
            prompt_config: Prompt configuration (uses default if None)
        """
        self.llm = llm_interface
        self.prompt_config = prompt_config or PromptConfig()
        self.all_themes: Set[str] = set()
        self.theme_counts: Counter = Counter()
    
    def extract_themes_from_text(self, text: str) -> List[str]:
        """Extract themes from a single text using LLM."""
        prompts = self.prompt_config.get_theme_extraction_prompts()
        
        system_prompt = prompts["system_prompt"]
        user_prompt = prompts["user_prompt"].format(text=text)
        
        response = self.llm.generate_response(system_prompt, user_prompt)
        
        # Parse the response to extract themes
        themes = self._parse_themes_from_response(response)
        return themes
    
    def extract_themes_from_documents(self, documents: List[str]) -> Dict[str, List[str]]:
        """
        Extract themes from multiple documents.
        
        Args:
            documents: List of document strings
            
        Returns:
            Dictionary mapping document index to list of themes
        """
        document_themes = {}
        
        for i, doc in enumerate(documents):
            print(f"Processing document {i+1}/{len(documents)}...")
            themes = self.extract_themes_from_text(doc)
            document_themes[str(i)] = themes
            
            # Add to master theme list
            self.all_themes.update(themes)
        
        return document_themes
    
    def count_theme_occurrences(self, documents: List[str], 
                              themes: List[str] = None) -> Dict[str, int]:
        """
        Count how many documents contain each theme.
        
        Args:
            documents: List of document strings
            themes: List of themes to check for (uses all discovered themes if None)
            
        Returns:
            Dictionary mapping themes to occurrence counts
        """
        if themes is None:
            themes = list(self.all_themes)
        
        theme_counts = Counter()
        prompts = self.prompt_config.get_theme_matching_prompts()
        
        for i, doc in enumerate(documents):
            print(f"Analyzing themes in document {i+1}/{len(documents)}...")
            
            # Get themes present in this document
            system_prompt = prompts["system_prompt"]
            user_prompt = prompts["user_prompt"].format(
                text=doc,
                themes="\n".join(f"- {theme}" for theme in themes)
            )
            
            response = self.llm.generate_response(system_prompt, user_prompt)
            present_themes = self._parse_themes_from_response(response, themes)
            
            # Update counts
            for theme in present_themes:
                theme_counts[theme] += 1
        
        self.theme_counts = theme_counts
        return dict(theme_counts)
    
    def _parse_themes_from_response(self, response: str, 
                                  valid_themes: List[str] = None) -> List[str]:
        """Parse themes from LLM response."""
        themes = []
        
        # Try different parsing approaches
        lines = response.split('\n')
        for line in lines:
            line = line.strip()
            
            # Remove numbering, bullets, dashes
            cleaned_line = re.sub(r'^[\d\.\-\*\•\s]*', '', line)
            cleaned_line = cleaned_line.strip()
            
            if cleaned_line:
                themes.append(cleaned_line)
        
        # If we have valid themes to match against, filter accordingly
        if valid_themes:
            matched_themes = []
            response_lower = response.lower()
            for theme in valid_themes:
                if theme.lower() in response_lower:
                    matched_themes.append(theme)
            return matched_themes
        
        return [theme for theme in themes if theme]
    
    def get_theme_summary(self) -> Dict:
        """Get summary of theme analysis."""
        return {
            "total_themes": len(self.all_themes),
            "all_themes": list(self.all_themes),
            "theme_counts": dict(self.theme_counts),
            "most_common_themes": self.theme_counts.most_common(10)
        }