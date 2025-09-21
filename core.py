"""Main ThemeExtractor class that orchestrates the entire process."""

from typing import List, Dict, Optional
from .llm_interface import LLMInterface, OpenAIInterface
from .theme_analyzer import ThemeAnalyzer
from .visualizer import WordCloudGenerator
from .prompt_config import PromptConfig

class ThemeExtractor:
    """Main class for theme extraction and analysis workflow."""
    
    def __init__(self, 
                 llm_interface: LLMInterface = None,
                 prompt_config_path: str = None,
                 wordcloud_config: Dict = None):
        """
        Initialize ThemeExtractor.
        
        Args:
            llm_interface: LLM interface to use (defaults to OpenAI)
            prompt_config_path: Path to custom prompt configuration
            wordcloud_config: Configuration for word cloud generation
        """
        # Initialize components
        self.llm = llm_interface or OpenAIInterface()
        self.prompt_config = PromptConfig(prompt_config_path)
        self.analyzer = ThemeAnalyzer(self.llm, self.prompt_config)
        
        wc_config = wordcloud_config or {}
        self.visualizer = WordCloudGenerator(**wc_config)
        
        # Results storage
        self.documents = []
        self.document_themes = {}
        self.theme_counts = {}
    
    def process_documents(self, documents: List[str]) -> Dict[str, int]:
        """
        Complete workflow: extract themes and count occurrences.
        
        Args:
            documents: List of document strings
            
        Returns:
            Dictionary mapping themes to occurrence counts
        """
        print(f"Starting theme extraction for {len(documents)} documents...")
        
        # Store documents
        self.documents = documents
        
        # Step 1: Extract themes from all documents
        print("Step 1: Extracting themes from documents...")
        self.document_themes = self.analyzer.extract_themes_from_documents(documents)
        
        # Step 2: Count theme occurrences across all documents
        print("Step 2: Counting theme occurrences...")
        self.theme_counts = self.analyzer.count_theme_occurrences(documents)
        
        print(f"Analysis complete! Found {len(self.analyzer.all_themes)} unique themes.")
        return self.theme_counts
    
    def generate_visualizations(self, 
                              output_dir: str = ".",
                              wordcloud_filename: str = "theme_wordcloud.png",
                              barchart_filename: str = "theme_barchart.png",
                              show_plots: bool = True) -> None:
        """
        Generate visualizations of theme analysis results.
        
        Args:
            output_dir: Directory to save visualizations
            wordcloud_filename: Filename for word cloud
            barchart_filename: Filename for bar chart
            show_plots: Whether to display plots
        """
        if not self.theme_counts:
            raise ValueError("No theme analysis results. Run process_documents() first.")
        
        from pathlib import Path
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Generate word cloud
        wordcloud_path = output_path / wordcloud_filename if wordcloud_filename else None
        self.visualizer.generate_wordcloud(
            self.theme_counts,
            title="Document Themes Word Cloud",
            save_path=str(wordcloud_path) if wordcloud_path else None,
            show_plot=show_plots
        )
        
        # Generate bar chart
        barchart_path = output_path / barchart_filename if barchart_filename else None
        self.visualizer.generate_bar_chart(
            self.theme_counts,
            title="Most Frequent Themes",
            save_path=str(barchart_path) if barchart_path else None,
            show_plot=show_plots
        )
    
    def get_results(self) -> Dict:
        """Get complete analysis results."""
        return {
            "summary": self.analyzer.get_theme_summary(),
            "document_themes": self.document_themes,
            "theme_counts": self.theme_counts
        }
    
    def save_results(self, output_path: str) -> None:
        """Save analysis results to JSON file."""
        import json
        results = self.get_results()
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"Results saved to {output_path}")