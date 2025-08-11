"""
Document loader for text files.
Handles loading and basic preprocessing of .txt files from a directory.
"""

import os
from pathlib import Path
from typing import List, Dict, Optional
import logging

logger = logging.getLogger(__name__)


class Document:
    """Represents a single document with metadata."""
    
    def __init__(self, content: str, filename: str, filepath: str):
        self.content = content
        self.filename = filename
        self.filepath = filepath
        self.word_count = len(content.split())
    
    def __repr__(self):
        return f"Document(filename='{self.filename}', words={self.word_count})"


class DocumentLoader:
    """Loads and manages text documents from a directory."""
    
    def __init__(self, input_dir: str, encoding: str = 'utf-8'):
        self.input_dir = Path(input_dir)
        self.encoding = encoding
        self.documents: List[Document] = []
    
    def load_documents(self, file_pattern: str = "*.txt") -> List[Document]:
        """
        Load all text files matching the pattern from input directory.
        
        Args:
            file_pattern: Glob pattern for files to load (default: "*.txt")
            
        Returns:
            List of Document objects
        """
        if not self.input_dir.exists():
            raise FileNotFoundError(f"Input directory not found: {self.input_dir}")
        
        txt_files = list(self.input_dir.glob(file_pattern))
        
        if not txt_files:
            logger.warning(f"No files matching '{file_pattern}' found in {self.input_dir}")
            return []
        
        documents = []
        for file_path in txt_files:
            try:
                document = self._load_single_file(file_path)
                if document:
                    documents.append(document)
            except Exception as e:
                logger.error(f"Failed to load {file_path}: {e}")
                continue
        
        self.documents = documents
        logger.info(f"Successfully loaded {len(documents)} documents")
        return documents
    
    def _load_single_file(self, file_path: Path) -> Optional[Document]:
        """Load a single text file."""
        try:
            with open(file_path, 'r', encoding=self.encoding) as f:
                content = f.read().strip()
            
            if not content:
                logger.warning(f"Empty file skipped: {file_path}")
                return None
            
            return Document(
                content=content,
                filename=file_path.name,
                filepath=str(file_path)
            )
            
        except UnicodeDecodeError:
            logger.error(f"Encoding error in file: {file_path}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error loading {file_path}: {e}")
            return None
    
    def get_documents(self) -> List[Document]:
        """Get loaded documents."""
        return self.documents
    
    def get_document_by_filename(self, filename: str) -> Optional[Document]:
        """Get a specific document by filename."""
        for doc in self.documents:
            if doc.filename == filename:
                return doc
        return None
    
    def get_summary(self) -> Dict[str, int]:
        """Get summary statistics of loaded documents."""
        total_words = sum(doc.word_count for doc in self.documents)
        return {
            "total_documents": len(self.documents),
            "total_words": total_words,
            "average_words_per_doc": total_words // len(self.documents) if self.documents else 0
        }


# Example usage
if __name__ == "__main__":
    loader = DocumentLoader("data/input")
    documents = loader.load_documents()
    
    print(f"Loaded {len(documents)} documents:")
    for doc in documents:
        print(f"  - {doc}")
    
    print("\nSummary:", loader.get_summary())