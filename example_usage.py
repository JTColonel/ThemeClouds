"""Example usage of the theme_extractor package."""

import os
from theme_extractor import ThemeExtractor
from theme_extractor.llm_interface import OpenAIInterface
from theme_extractor.utils import create_sample_config

def main():
    """Example usage of the theme extractor."""
    
    # Sample documents for testing
    sample_documents = [
        """The rapid advancement of artificial intelligence has transformed various industries. 
        Machine learning algorithms are now capable of processing vast amounts of data and 
        making predictions with unprecedented accuracy. However, concerns about job displacement 
        and ethical implications of AI continue to grow among policymakers and the general public.""",
        
        """Climate change remains one of the most pressing challenges of our time. Rising global 
        temperatures, melting ice caps, and extreme weather events are becoming more frequent. 
        Governments worldwide are implementing policies to reduce carbon emissions and transition 
        to renewable energy sources. The Paris Agreement represents a significant step toward 
        international cooperation on environmental issues.""",
        
        """The COVID-19 pandemic has accelerated the adoption of remote work technologies. 
        Video conferencing platforms, cloud computing, and collaboration tools have become 
        essential for maintaining business operations. This shift has also highlighted the 
        importance of work-life balance and mental health in the modern workplace.""",
        
        """Blockchain technology and cryptocurrencies are revolutionizing the financial sector. 
        Decentralized finance (DeFi) platforms offer new ways to lend, borrow, and invest without 
        traditional intermediaries. However, regulatory uncertainty and market volatility continue 
        to pose challenges for widespread adoption.""",
        
        """Social media platforms have fundamentally changed how people communicate and consume 
        information. While these platforms enable global connectivity and information sharing, 
        they also face criticism for their role in spreading misinformation and creating echo 
        chambers. Content moderation and digital privacy remain ongoing concerns.""",
        
        """Healthcare systems worldwide are undergoing digital transformation. Electronic health 
        records, telemedicine, and AI-powered diagnostic tools are improving patient care and 
        operational efficiency. The pandemic has particularly accelerated the adoption of 
        telehealth services, making healthcare more accessible to remote populations."""
    ]
    
    # Example 1: Basic usage with default configuration
    print("=== Example 1: Basic Usage ===")
    
    # Initialize with OpenAI (requires OPENAI_API_KEY environment variable)
    extractor = ThemeExtractor()
    
    # Process documents
    try:
        theme_counts = extractor.process_documents(sample_documents)
        
        # Print results
        print(f"\nFound {len(theme_counts)} themes:")
        for theme, count in sorted(theme_counts.items(), key=lambda x: x[1], reverse=True):
            print(f"  {theme}: {count}")
        
        # Generate visualizations
        extractor.generate_visualizations(
            output_dir="output",
            show_plots=False  # Set to True to display plots
        )
        
        # Save results
        extractor.save_results("output/theme_analysis_results.json")
        
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure to set your OPENAI_API_KEY environment variable")
    
    # Example 2: Custom configuration
    print("\n=== Example 2: Custom Configuration ===")
    
    # Create custom prompt configuration
    create_sample_config("custom_prompts.json")
    
    # Initialize with custom configuration
    custom_extractor = ThemeExtractor(
        prompt_config_path="custom_prompts.json",
        wordcloud_config={
            "width": 1200,
            "height": 600,
            "background_color": "black",
            "colormap": "plasma"
        }
    )
    
    print("Custom extractor initialized with custom prompts and wordcloud settings")
    
    # Example 3: Using different LLM interfaces
    print("\n=== Example 3: Custom LLM Interface ===")
    
    # Custom OpenAI configuration
    openai_interface = OpenAIInterface(
        model="gpt-4",  # Use GPT-4 instead of default
        max_retries=5,
        delay_between_calls=2.0
    )
    
    advanced_extractor = ThemeExtractor(
        llm_interface=openai_interface,
        wordcloud_config={"colormap": "viridis"}
    )
    
    print("Advanced extractor with GPT-4 configured")

if __name__ == "__main__":
    main()