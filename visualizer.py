from typing import Dict, Optional, Tuple
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from collections import Counter

class WordCloudGenerator:
    """Generates word clouds from theme frequency data."""
    
    def __init__(self, width: int = 800, height: int = 400, 
                 background_color: str = 'white', colormap: str = 'viridis'):
        """
        Initialize word cloud generator.
        
        Args:
            width: Width of the word cloud
            height: Height of the word cloud
            background_color: Background color
            colormap: Matplotlib colormap name
        """
        self.width = width
        self.height = height
        self.background_color = background_color
        self.colormap = colormap
    
    def generate_wordcloud(self, theme_counts: Dict[str, int], 
                          title: str = "Theme Frequency",
                          save_path: str = None,
                          show_plot: bool = True) -> WordCloud:
        """
        Generate word cloud from theme frequency data.
        
        Args:
            theme_counts: Dictionary mapping themes to frequency counts
            title: Title for the plot
            save_path: Path to save the image (optional)
            show_plot: Whether to display the plot
            
        Returns:
            WordCloud object
        """
        if not theme_counts:
            raise ValueError("No theme counts provided")
        
        # Create WordCloud
        wordcloud = WordCloud(
            width=self.width,
            height=self.height,
            background_color=self.background_color,
            colormap=self.colormap,
            max_words=100,
            relative_scaling=0.5,
            random_state=42
        ).generate_from_frequencies(theme_counts)
        
        if show_plot or save_path:
            plt.figure(figsize=(12, 6))
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis('off')
            plt.title(title, fontsize=16, fontweight='bold')
            
            if save_path:
                plt.savefig(save_path, bbox_inches='tight', dpi=300)
            
            if show_plot:
                plt.show()
        
        return wordcloud
    
    def generate_bar_chart(self, theme_counts: Dict[str, int],
                          top_n: int = 15,
                          title: str = "Top Themes by Frequency",
                          save_path: str = None,
                          show_plot: bool = True) -> None:
        """
        Generate horizontal bar chart of theme frequencies.
        
        Args:
            theme_counts: Dictionary mapping themes to frequency counts
            top_n: Number of top themes to show
            title: Title for the plot
            save_path: Path to save the image (optional)
            show_plot: Whether to display the plot
        """
        if not theme_counts:
            raise ValueError("No theme counts provided")
        
        # Get top themes
        counter = Counter(theme_counts)
        top_themes = counter.most_common(top_n)
        
        themes = [item[0] for item in top_themes]
        counts = [item[1] for item in top_themes]
        
        plt.figure(figsize=(10, max(6, len(themes) * 0.3)))
        bars = plt.barh(range(len(themes)), counts)
        plt.yticks(range(len(themes)), themes)
        plt.xlabel('Frequency')
        plt.title(title)
        plt.gca().invert_yaxis()  # Highest at top
        
        # Add value labels on bars
        for i, (bar, count) in enumerate(zip(bars, counts)):
            plt.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2, 
                    str(count), ha='left', va='center')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
        
        if show_plot:
            plt.show()