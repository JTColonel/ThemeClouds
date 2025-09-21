"""Interface for LLM communication. Slight change"""

from abc import ABC, abstractmethod
from typing import List, Optional
import openai
import time

class LLMInterface(ABC):
    """Abstract base class for LLM interfaces."""
    
    @abstractmethod
    def generate_response(self, system_prompt: str, user_prompt: str) -> str:
        """Generate a response from the LLM."""
        pass

class OpenAIInterface(LLMInterface):
    """OpenAI API interface."""
    
    def __init__(self, api_key: str = None, model: str = "gpt-3.5-turbo", 
                 max_retries: int = 3, delay_between_calls: float = 1.0):
        """
        Initialize OpenAI interface.
        
        Args:
            api_key: OpenAI API key (if None, expects OPENAI_API_KEY env var)
            model: Model to use
            max_retries: Maximum number of retry attempts
            delay_between_calls: Delay between API calls to avoid rate limiting
        """
        if api_key:
            openai.api_key = api_key
        self.model = model
        self.max_retries = max_retries
        self.delay_between_calls = delay_between_calls
    
    def generate_response(self, system_prompt: str, user_prompt: str) -> str:
        """Generate response using OpenAI API."""
        for attempt in range(self.max_retries):
            try:
                response = openai.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=0.3,
                    max_tokens=1000
                )
                
                time.sleep(self.delay_between_calls)
                return response.choices[0].message.content.strip()
                
            except Exception as e:
                if attempt == self.max_retries - 1:
                    raise e
                time.sleep(2 ** attempt)  # Exponential backoff
        
        return ""

class ClaudeInterface(LLMInterface):
    """Anthropic Claude interface (placeholder - implement based on your API access)."""
    
    def __init__(self, api_key: str = None, model: str = "claude-3-sonnet-20240229"):
        """Initialize Claude interface."""
        self.api_key = api_key
        self.model = model
    
    def generate_response(self, system_prompt: str, user_prompt: str) -> str:
        """Generate response using Claude API."""
        # Implement based on Anthropic's API
        # This is a placeholder
        raise NotImplementedError("Claude interface needs to be implemented with actual API calls")
