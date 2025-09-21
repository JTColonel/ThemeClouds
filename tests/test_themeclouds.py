"""Tests for the theme extractor package."""

import pytest
from unittest.mock import Mock, patch
from theme_extractor import ThemeExtractor
from theme_extractor.llm_interface import LLMInterface
from theme_extractor.prompt_config import PromptConfig
from theme_extractor.theme_analyzer import ThemeAnalyzer

class MockLLMInterface(LLMInterface):
    """Mock LLM interface for testing."""
    
    def __init__(self, responses=None):
        self.responses = responses or [
            "1. Technology\n2. Innovation\n3. Digital transformation",
            "Technology, Innovation",
        ]
        self.call_count = 0
    
    def generate_response(self, system_prompt: str, user_prompt: str) -> str:
        response = self.responses[self.call_count % len(self.responses)]
        self.call_count += 1
        return response

class TestPromptConfig:
    """Test prompt configuration functionality."""
    
    def test_default_prompts_loaded(self):
        config = PromptConfig()
        assert "theme_extraction" in config.prompts
        assert "theme_matching" in config.prompts
    
    def test_get_prompts(self):
        config = PromptConfig()
        extraction_prompts = config.get_theme_extraction_prompts()
        
        assert "system_prompt" in extraction_prompts
        assert "user_prompt" in extraction_prompts
        assert len(extraction_prompts["system_prompt"]) > 0

class TestThemeAnalyzer:
    """Test theme analysis functionality."""
    
    def test_extract_themes_from_text(self):
        mock_llm = MockLLMInterface()
        analyzer = ThemeAnalyzer(mock_llm)
        
        themes = analyzer.extract_themes_from_text("Sample text about technology")
        
        assert len(themes) > 0
        assert "Technology" in themes
        assert "Innovation" in themes
    
    def test_extract_themes_from_documents(self):
        mock_llm = MockLLMInterface()
        analyzer = ThemeAnalyzer(mock_llm)
        
        documents = [
            "Text about technology and innovation",
            "Another document about digital transformation"
        ]
        
        result = analyzer.extract_themes_from_documents(documents)
        
        assert len(result) == 2
        assert "0" in result
        assert "1" in result
        assert len(analyzer.all_themes) > 0

class TestThemeExtractor:
    """Test main theme extractor functionality."""
    
    def test_initialization(self):
        mock_llm = MockLLMInterface()
        extractor = ThemeExtractor(llm_interface=mock_llm)
        
        assert extractor.llm is not None
        assert extractor.analyzer is not None
        assert extractor.visualizer is not None
    
    def test_process_documents(self):
        mock_llm = MockLLMInterface([
            "1. Technology\n2. Innovation",
            "Technology",
            "Innovation",
        ])
        
        extractor = ThemeExtractor(llm_interface=mock_llm)
        documents = ["Sample tech document", "Another innovation document"]
        
        theme_counts = extractor.process_documents(documents)
        
        assert isinstance(theme_counts, dict)
        assert len(theme_counts) > 0
    
    def test_get_results(self):
        mock_llm = MockLLMInterface()
        extractor = ThemeExtractor(llm_interface=mock_llm)
        
        # Process some documents first
        documents = ["Test document"]
        extractor.process_documents(documents)
        
        results = extractor.get_results()
        
        assert "summary" in results
        assert "document_themes" in results
        assert "theme_counts" in results

if __name__ == "__main__":
    pytest.main([__file__])