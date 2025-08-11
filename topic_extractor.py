import re
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import logging
from pathlib import Path
import json

from .document_loader import Document
from .llm_client import LlamaClient, GenerationParams
from .config_loader import ConfigLoader, PromptConfig

logger = logging.getLogger(__name__)


@dataclass
class TopicExtractionResult:
    """Result of topic extraction for a single document or collection."""
    document_filename: str
    topics: List[str]
    raw_response: str
    extraction_prompt: str
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'document_filename': self.document_filename,
            'topics': self.topics,
            'raw_response': self.raw_response,
            'extraction_prompt': self.extraction_prompt
        }


class TopicExtractor:
    """Extracts topics and keywords from documents using LLM analysis."""
    
    def __init__(self, llm_client: LlamaClient, config_loader: Optional[ConfigLoader] = None):
        self.llm_client = llm_client
        self.config_loader = config_loader or ConfigLoader()
        self.default_generation_params = GenerationParams(
            max_new_tokens=800,
            temperature=0.3,
            do_sample=False
        )
    
    def extract_topics_from_collection(
        self,
        documents: List[Document],
        prompt_name: str,
        num_topics: Optional[int] = None,
        generation_params: Optional[GenerationParams] = None
    ) -> List[str]:
        """
        Extract N topics that appear across a collection of documents.
        
        Args:
            documents: List of documents to analyze
            prompt_name: Name of the prompt to load from configuration
            num_topics: Number of topics to extract (uses prompt default if None)
            generation_params: LLM generation parameters
            
        Returns:
            List of extracted topics
        """
        if not documents:
            raise ValueError("No documents provided")
        
        # Load prompt configuration
        prompt_config = self.config_loader.get_prompt(prompt_name)
        if not prompt_config:
            raise ValueError(f"Prompt '{prompt_name}' not found in configuration")
        
        # Use prompt default if num_topics not specified
        if num_topics is None:
            num_topics = prompt_config.default_num_topics
        
        # Combine all documents for global topic extraction
        combined_content = "\n\n".join([
            f"=== {doc.filename} ===\n{doc.content}" 
            for doc in documents
        ])
        
        # Format the prompt
        formatted_prompt = prompt_config.template.format(
            num_topics=num_topics,
            content=combined_content
        )
        
        logger.info(f"Extracting {num_topics} topics from {len(documents)} documents using prompt '{prompt_name}'")
        
        params = generation_params or self.default_generation_params
        raw_response = self.llm_client.generate(
            system_msg=prompt_config.system_message,
            user_msg=formatted_prompt,
            generation_params=params
        )
        
        # Parse topics from response
        topics = self._parse_topic_list(raw_response)
        
        logger.info(f"Extracted {len(topics)} topics: {topics[:3]}..." if len(topics) > 3 else f"Extracted topics: {topics}")
        
        return topics
    
    def extract_topics_per_document(
        self,
        documents: List[Document],
        prompt_name: str,
        num_topics: Optional[int] = None,
        generation_params: Optional[GenerationParams] = None
    ) -> List[TopicExtractionResult]:
        """
        Extract topics from each document individually.
        
        Args:
            documents: List of documents to analyze
            prompt_name: Name of the prompt to load from configuration
            num_topics: Number of topics to extract per document (uses prompt default if None)
            generation_params: LLM generation parameters
            
        Returns:
            List of TopicExtractionResult objects
        """
        if not documents:
            raise ValueError("No documents provided")
        
        # Load prompt configuration
        prompt_config = self.config_loader.get_prompt(prompt_name)
        if not prompt_config:
            raise ValueError(f"Prompt '{prompt_name}' not found in configuration")
        
        # Use prompt default if num_topics not specified
        if num_topics is None:
            num_topics = prompt_config.default_num_topics
        
        results = []
        params = generation_params or self.default_generation_params
        
        for i, doc in enumerate(documents):
            logger.info(f"Processing document {i+1}/{len(documents)}: {doc.filename}")
            
            # Format the prompt for this specific document
            formatted_prompt = prompt_config.template.format(
                num_topics=num_topics,
                content=doc.content,
                filename=doc.filename
            )
            
            try:
                raw_response = self.llm_client.generate(
                    system_msg=prompt_config.system_message,
                    user_msg=formatted_prompt,
                    generation_params=params
                )
                
                # Parse topics from response
                topics = self._parse_topic_list(raw_response)
                
                result = TopicExtractionResult(
                    document_filename=doc.filename,
                    topics=topics,
                    raw_response=raw_response,
                    extraction_prompt=formatted_prompt
                )
                
                results.append(result)
                logger.info(f"Extracted {len(topics)} topics from {doc.filename}")
                
            except Exception as e:
                logger.error(f"Failed to extract topics from {doc.filename}: {e}")
                # Create empty result for failed extractions
                result = TopicExtractionResult(
                    document_filename=doc.filename,
                    topics=[],
                    raw_response=f"ERROR: {str(e)}",
                    extraction_prompt=formatted_prompt
                )
                results.append(result)
        
        return results
    
    def _parse_topic_list(self, raw_response: str) -> List[str]:
        """
        Parse topics from LLM response.
        Handles various formats like bullet points, numbered lists, etc.
        """
        topics = []
        lines = raw_response.strip().split('\n')
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Handle different list formats
            # Bullet points: - topic, * topic, • topic
            bullet_match = re.match(r'^[-*•]\s*(.+)$', line)
            if bullet_match:
                topic = bullet_match.group(1).strip()
                topics.append(self._clean_topic(topic))
                continue
            
            # Numbered lists: 1. topic, 1) topic
            number_match = re.match(r'^\d+[.)]\s*(.+)$', line)
            if number_match:
                topic = number_match.group(1).strip()
                topics.append(self._clean_topic(topic))
                continue
            
            # Section headers (like ### insta)
            section_match = re.match(r'^#+\s*(.+)$', line)
            if section_match:
                continue  # Skip section headers
            
            # Plain line (if it looks like a topic)
            if len(line.split()) <= 5 and not line.endswith(':'):
                topics.append(self._clean_topic(line))
        
        return topics
    
    def _clean_topic(self, topic: str) -> str:
        """Clean and normalize a topic string."""
        # Remove quotes, extra whitespace
        topic = re.sub(r'^["\']|["\']$', '', topic)
        topic = topic.strip()
        
        # Remove trailing punctuation except for meaningful ones
        topic = re.sub(r'[.,:;!?]+$', '', topic)
        
        return topic
    
    def save_results(self, results: List[TopicExtractionResult], output_path: str):
        """Save extraction results to JSON file."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            'extraction_timestamp': str(pd.Timestamp.now()),
            'total_documents': len(results),
            'results': [result.to_dict() for result in results]
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved extraction results to {output_path}")
    
    def load_results(self, input_path: str) -> List[TopicExtractionResult]:
        """Load extraction results from JSON file."""
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        results = []
        for item in data['results']:
            result = TopicExtractionResult(
                document_filename=item['document_filename'],
                topics=item['topics'],
                raw_response=item['raw_response'],
                extraction_prompt=item['extraction_prompt']
            )
            results.append(result)
        
    def get_available_prompts(self) -> List[str]:
        """Get list of available prompt names."""
        return self.config_loader.get_prompt_names()
    
    def get_prompt_info(self, prompt_name: str) -> Optional[PromptConfig]:
        """Get information about a specific prompt."""
        return self.config_loader.get_prompt(prompt_name)


# Example usage and prompt templates
WEBCAM_EXPERIENCE_PROMPT = """
You are analyzing interview transcripts where participants were asked to share their experiences using different webcam setups.

Your task is to identify **exactly {num_topics} meaningful and distinctive words or short phrases** that summarize participants' real experiences.

Guidelines:
- Do NOT just pick the most frequent words.
- Select words or short phrases that are **emotionally descriptive**, **technically relevant**, or **highlight distinctive qualities** (positive or negative).
- Avoid: generic words (e.g., "thing", "camera"), filler words, or phrases repeated from questions.
- Focus only on participant speech that offers insight, reaction, or description.

Return exactly {num_topics} high-quality descriptors as a bullet list.

Content to analyze:
{content}
"""

GENERAL_TOPICS_PROMPT = """
Analyze the following documents and identify the {num_topics} most important topics or themes that appear across the content.

Focus on:
- Key concepts and ideas
- Important themes
- Significant subjects discussed
- Notable patterns or recurring elements

Return exactly {num_topics} topics as a simple bullet list.

Content to analyze:
{content}
"""


if __name__ == "__main__":
    # Example usage
    from .llm_client import LlamaClient
    from .document_loader import DocumentLoader
    
    # Setup
    llm_client = LlamaClient(access_token="your_token")
    llm_client.load_model()
    
    loader = DocumentLoader("data/input")
    documents = loader.load_documents()
    
    extractor = TopicExtractor(llm_client)
    
    # Extract topics per document
    results = extractor.extract_topics_per_document(
        documents=documents,
        prompt_template=WEBCAM_EXPERIENCE_PROMPT,
        num_topics=10
    )
    
    # Save results
    extractor.save_results(results, "data/output/topics/extraction_results.json")