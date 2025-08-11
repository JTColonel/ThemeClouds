"""
LLM client for Hugging Face Llama models.
Handles model loading, prompt formatting, and text generation.
"""

import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    GenerationConfig,
    BitsAndBytesConfig
)
from huggingface_hub import login
import logging
from typing import Optional, Dict, Any
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class GenerationParams:
    """Parameters for text generation."""
    max_new_tokens: int = 500
    temperature: float = 0.3
    do_sample: bool = False
    top_p: float = 1.0
    repetition_penalty: float = 1.0


class LlamaClient:
    """Client for interacting with Llama models via Hugging Face."""
    
    def __init__(
        self, 
        model_id: str = "meta-llama/Llama-3.3-70B-Instruct",
        access_token: Optional[str] = None,
        use_4bit_quantization: bool = True,
        device_map: str = "auto"
    ):
        self.model_id = model_id
        self.access_token = access_token
        self.use_4bit_quantization = use_4bit_quantization
        self.device_map = device_map
        
        self.tokenizer = None
        self.model = None
        self.is_loaded = False
    
    def load_model(self):
        """Load the model and tokenizer."""
        if self.is_loaded:
            logger.info("Model already loaded")
            return
        
        try:
            # Login to Hugging Face if token provided
            if self.access_token:
                login(token=self.access_token)
            
            logger.info(f"Loading tokenizer for {self.model_id}")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_id, 
                token=self.access_token
            )
            
            # Configure quantization if requested
            quantization_config = None
            if self.use_4bit_quantization:
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True, 
                    bnb_4bit_compute_dtype=torch.float16
                )
                logger.info("Using 4-bit quantization")
            
            logger.info(f"Loading model {self.model_id}")
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                quantization_config=quantization_config,
                device_map=self.device_map,
                torch_dtype=torch.float16,
                token=self.access_token
            )
            
            self.is_loaded = True
            logger.info("Model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def build_prompt(self, system_msg: str, user_msg: str) -> str:
        """
        Build a formatted prompt using the chat template.
        
        Args:
            system_msg: System message/instructions
            user_msg: User message/content
            
        Returns:
            Formatted prompt string
        """
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ]
        
        return self.tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
    
    def generate(
        self, 
        system_msg: str, 
        user_msg: str, 
        generation_params: Optional[GenerationParams] = None
    ) -> str:
        """
        Generate text using the loaded model.
        
        Args:
            system_msg: System message/instructions
            user_msg: User message/content
            generation_params: Generation parameters
            
        Returns:
            Generated text
        """
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        if generation_params is None:
            generation_params = GenerationParams()
        
        # Build prompt
        formatted_prompt = self.build_prompt(system_msg, user_msg)
        
        # Tokenize input
        inputs = self.tokenizer(
            formatted_prompt, 
            return_tensors="pt"
        ).to(self.model.device)
        
        # Configure generation
        generation_config = GenerationConfig(
            max_new_tokens=generation_params.max_new_tokens,
            temperature=generation_params.temperature,
            do_sample=generation_params.do_sample,
            top_p=generation_params.top_p,
            repetition_penalty=generation_params.repetition_penalty,
        )
        
        # Generate
        try:
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs, 
                    generation_config=generation_config
                )
            
            # Decode only the new tokens
            input_length = inputs['input_ids'].shape[1]
            result = self.tokenizer.decode(
                outputs[0][input_length:], 
                skip_special_tokens=True
            )
            
            return result.strip()
            
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            raise
    
    def summarize_document(
        self, 
        prompt: str, 
        document_content: str, 
        generation_params: Optional[GenerationParams] = None
    ) -> str:
        """
        Convenience method for document summarization.
        
        Args:
            prompt: The summarization prompt/instructions
            document_content: Content of the document to summarize
            generation_params: Generation parameters
            
        Returns:
            Summary text
        """
        user_message = f"Document: {document_content}"
        return self.generate(prompt, user_message, generation_params)
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model."""
        return {
            "model_id": self.model_id,
            "is_loaded": self.is_loaded,
            "device": str(self.model.device) if self.model else None,
            "quantization": self.use_4bit_quantization
        }


# Example usage
if __name__ == "__main__":
    # Initialize client
    client = LlamaClient(access_token="your_token_here")
    
    # Load model
    client.load_model()
    
    # Generate text
    result = client.generate(
        system_msg="You are a helpful assistant that summarizes documents.",
        user_msg="Document: This is a sample document about AI.",
        generation_params=GenerationParams(max_new_tokens=100)
    )
    
    print("Generated:", result)
    print("Model info:", client.get_model_info())