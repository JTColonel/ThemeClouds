"""Interface for LLM communication. Slight change"""

from abc import ABC, abstractmethod
from typing import List, Optional
from llama_cpp import Llama
import time

class LLMInterface(ABC):
    """Abstract base class for LLM interfaces."""
    
    @abstractmethod
    def generate_response(self, system_prompt: str, user_prompt: str) -> str:
        """Generate a response from the LLM."""
        pass


class LlamaCppInterface(LLMInterface):
    """Llama.cpp interface for local GGUF models."""
    
    def __init__(self, llm_path: str, max_retries: int = 3, 
                 delay_between_calls: float = 0.1, n_ctx: int = 4096,
                 temperature: float = 0.3, max_tokens: int = 1000,
                 n_threads: int = None, verbose: bool = False):
        """
        Initialize Llama.cpp interface.
        
        Args:
            llm_path: Path to the GGUF model file
            max_retries: Maximum number of retry attempts
            delay_between_calls: Delay between calls (usually shorter for local models)
            n_ctx: Context window size
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            n_threads: Number of threads to use (None for auto)
            verbose: Whether to enable verbose logging
        """
        
        self.llm_path = llm_path
        self.max_retries = max_retries
        self.delay_between_calls = delay_between_calls
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        # Initialize the Llama model
        self.llm = Llama(
            model_path=llm_path,
            n_ctx=n_ctx,
            n_threads=n_threads,
            verbose=verbose
        )
    
    def generate_response(self, system_prompt: str, user_prompt: str) -> str:
        """Generate response using Llama.cpp."""
        # Format for Llama 3.3 Instruct
        formatted_prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>

{user_prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""
        
        for attempt in range(self.max_retries):
            try:
                response = self.llm(
                    formatted_prompt,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                    stop=["<|eot_id|>", "<|end_of_text|>"],
                    echo=False
                )
                
                time.sleep(self.delay_between_calls)
                
                # Extract the generated text
                generated_text = response['choices'][0]['text'].strip()
                return generated_text
                
            except Exception as e:
                if attempt == self.max_retries - 1:
                    raise e
                time.sleep(2 ** attempt)  # Exponential backoff
        
        return ""

