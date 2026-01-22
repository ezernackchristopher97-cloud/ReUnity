"""
Document Chunker

Splits documents into chunks for RAG indexing.

DISCLAIMER: This is not a clinical or treatment tool.

Author: Christopher Ezernack
"""

from __future__ import annotations

import hashlib
import re
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class Chunk:
    """A chunk of text from a document."""
    
    id: str
    text: str
    source: str
    start_char: int
    end_char: int
    metadata: dict[str, Any] = field(default_factory=dict)
    
    @property
    def hash(self) -> str:
        """Content hash for deduplication."""
        return hashlib.md5(self.text.encode()).hexdigest()


class DocumentChunker:
    """
    Splits documents into overlapping chunks for RAG.
    
    Supports multiple chunking strategies:
    - Fixed size with overlap
    - Sentence-based
    - Paragraph-based
    """
    
    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        min_chunk_size: int = 50,
        strategy: str = "fixed",
        respect_sentences: bool = True,
    ) -> None:
        """
        Initialize the chunker.
        
        Args:
            chunk_size: Target chunk size in characters.
            chunk_overlap: Overlap between chunks.
            min_chunk_size: Minimum chunk size to keep.
            strategy: Chunking strategy ("fixed", "sentence", "paragraph").
            respect_sentences: Try not to break mid-sentence.
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size
        self.strategy = strategy
        self.respect_sentences = respect_sentences
    
    def chunk_text(
        self,
        text: str,
        source: str = "unknown",
        metadata: dict[str, Any] | None = None,
    ) -> list[Chunk]:
        """
        Chunk a text string.
        
        Args:
            text: Text to chunk.
            source: Source identifier.
            metadata: Additional metadata for chunks.
        
        Returns:
            List of Chunk objects.
        """
        metadata = metadata or {}
        
        if self.strategy == "sentence":
            return self._chunk_by_sentences(text, source, metadata)
        elif self.strategy == "paragraph":
            return self._chunk_by_paragraphs(text, source, metadata)
        else:
            return self._chunk_fixed(text, source, metadata)
    
    def chunk_file(
        self,
        file_path: str | Path,
        metadata: dict[str, Any] | None = None,
    ) -> list[Chunk]:
        """
        Chunk a file.
        
        Args:
            file_path: Path to file.
            metadata: Additional metadata.
        
        Returns:
            List of Chunk objects.
        """
        path = Path(file_path)
        
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        
        text = path.read_text(encoding="utf-8")
        
        file_metadata = {
            "filename": path.name,
            "filepath": str(path),
            **(metadata or {}),
        }
        
        return self.chunk_text(text, source=str(path), metadata=file_metadata)
    
    def chunk_directory(
        self,
        dir_path: str | Path,
        extensions: list[str] | None = None,
        recursive: bool = True,
    ) -> list[Chunk]:
        """
        Chunk all files in a directory.
        
        Args:
            dir_path: Directory path.
            extensions: File extensions to include (e.g., [".txt", ".md"]).
            recursive: Search subdirectories.
        
        Returns:
            List of Chunk objects from all files.
        """
        path = Path(dir_path)
        extensions = extensions or [".txt", ".md", ".rst"]
        
        all_chunks = []
        
        pattern = "**/*" if recursive else "*"
        for file_path in path.glob(pattern):
            if file_path.is_file() and file_path.suffix.lower() in extensions:
                try:
                    chunks = self.chunk_file(file_path)
                    all_chunks.extend(chunks)
                except Exception as e:
                    print(f"Warning: Failed to chunk {file_path}: {e}")
        
        return all_chunks
    
    def _chunk_fixed(
        self,
        text: str,
        source: str,
        metadata: dict[str, Any],
    ) -> list[Chunk]:
        """Fixed-size chunking with overlap."""
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + self.chunk_size
            
            # Adjust end to respect sentence boundaries
            if self.respect_sentences and end < len(text):
                # Look for sentence end near the boundary
                search_start = max(start + self.min_chunk_size, end - 100)
                search_end = min(len(text), end + 100)
                search_text = text[search_start:search_end]
                
                # Find last sentence boundary
                sentence_ends = [
                    m.end() for m in re.finditer(r'[.!?]\s+', search_text)
                ]
                
                if sentence_ends:
                    # Use the last sentence boundary in range
                    best_end = search_start + sentence_ends[-1]
                    if best_end > start + self.min_chunk_size:
                        end = best_end
            
            # Extract chunk text
            chunk_text = text[start:end].strip()
            
            if len(chunk_text) >= self.min_chunk_size:
                chunks.append(Chunk(
                    id=str(uuid.uuid4()),
                    text=chunk_text,
                    source=source,
                    start_char=start,
                    end_char=end,
                    metadata={
                        **metadata,
                        "chunk_index": len(chunks),
                    },
                ))
            
            # Move start with overlap
            start = end - self.chunk_overlap
            if start >= len(text) - self.min_chunk_size:
                break
        
        return chunks
    
    def _chunk_by_sentences(
        self,
        text: str,
        source: str,
        metadata: dict[str, Any],
    ) -> list[Chunk]:
        """Sentence-based chunking."""
        # Split into sentences
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        chunks = []
        current_chunk = []
        current_length = 0
        start_char = 0
        
        for sentence in sentences:
            sentence_len = len(sentence)
            
            if current_length + sentence_len > self.chunk_size and current_chunk:
                # Save current chunk
                chunk_text = " ".join(current_chunk)
                chunks.append(Chunk(
                    id=str(uuid.uuid4()),
                    text=chunk_text,
                    source=source,
                    start_char=start_char,
                    end_char=start_char + len(chunk_text),
                    metadata={
                        **metadata,
                        "chunk_index": len(chunks),
                    },
                ))
                
                # Start new chunk with overlap
                overlap_sentences = current_chunk[-2:] if len(current_chunk) > 2 else []
                current_chunk = overlap_sentences
                current_length = sum(len(s) for s in current_chunk)
                start_char = start_char + len(chunk_text) - current_length
            
            current_chunk.append(sentence)
            current_length += sentence_len
        
        # Save final chunk
        if current_chunk:
            chunk_text = " ".join(current_chunk)
            if len(chunk_text) >= self.min_chunk_size:
                chunks.append(Chunk(
                    id=str(uuid.uuid4()),
                    text=chunk_text,
                    source=source,
                    start_char=start_char,
                    end_char=start_char + len(chunk_text),
                    metadata={
                        **metadata,
                        "chunk_index": len(chunks),
                    },
                ))
        
        return chunks
    
    def _chunk_by_paragraphs(
        self,
        text: str,
        source: str,
        metadata: dict[str, Any],
    ) -> list[Chunk]:
        """Paragraph-based chunking."""
        # Split by double newlines
        paragraphs = re.split(r'\n\n+', text)
        
        chunks = []
        current_chunk = []
        current_length = 0
        start_char = 0
        
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            
            para_len = len(para)
            
            if current_length + para_len > self.chunk_size and current_chunk:
                # Save current chunk
                chunk_text = "\n\n".join(current_chunk)
                chunks.append(Chunk(
                    id=str(uuid.uuid4()),
                    text=chunk_text,
                    source=source,
                    start_char=start_char,
                    end_char=start_char + len(chunk_text),
                    metadata={
                        **metadata,
                        "chunk_index": len(chunks),
                    },
                ))
                
                # Start new chunk
                current_chunk = []
                current_length = 0
                start_char = start_char + len(chunk_text) + 2
            
            current_chunk.append(para)
            current_length += para_len
        
        # Save final chunk
        if current_chunk:
            chunk_text = "\n\n".join(current_chunk)
            if len(chunk_text) >= self.min_chunk_size:
                chunks.append(Chunk(
                    id=str(uuid.uuid4()),
                    text=chunk_text,
                    source=source,
                    start_char=start_char,
                    end_char=start_char + len(chunk_text),
                    metadata={
                        **metadata,
                        "chunk_index": len(chunks),
                    },
                ))
        
        return chunks
