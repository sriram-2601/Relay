"""
PDF Ingestion and Document Chunking Module for 'The Relay'.
Extracts text with structural metadata (section titles, page numbers, arXiv IDs)
from the 3 target research papers.
"""

import os
import re
from typing import List, Dict, Any
import fitz  # PyMuPDF

PAPERS_METADATA = {
    "1706.03762v7": {
        "title": "Attention Is All You Need",
        "short_title": "Attention",
        "filename": "data/papers/attention_1706.03762v7.pdf"
    },
    "2410.00037v1": {
        "title": "Moshi: a speech-text foundation model for real-time dialogue",
        "short_title": "Moshi",
        "filename": "data/papers/moshi_2410.00037v1.pdf"
    },
    "2507.07955v2": {
        "title": "Dynamic Chunking for End-to-End Hierarchical Sequence Modeling",
        "short_title": "Dynamic Chunking",
        "filename": "data/papers/dynamic_chunking_2507.07955v2.pdf"
    }
}

# Regex to detect numbered or common academic section headers
SECTION_REGEX = re.compile(
    r"^(?:(?:\d+\.?)+\s+[A-Z][A-Za-z0-9\s,\-:]{2,60}|Abstract|References|Introduction|Conclusion|Background|Architecture|Results|Discussion|Related Work|Experiments|Appendix(?:\s+[A-Z0-9])?)$",
    re.MULTILINE
)

def extract_pages(pdf_path: str) -> List[Dict[str, Any]]:
    """Extracts raw text page by page with basic block structure."""
    doc = fitz.open(pdf_path)
    pages = []
    current_section = "Introduction / Front Matter"

    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text("text")
        
        # Look for section headers on this page
        lines = [line.strip() for line in text.split("\n") if line.strip()]
        for line in lines:
            if SECTION_REGEX.match(line) and len(line) < 60:
                current_section = line
                break

        pages.append({
            "page_num": page_num + 1,
            "section": current_section,
            "text": text
        })
    doc.close()
    return pages

def chunk_text(text: str, chunk_size_words: int, overlap_words: int) -> List[str]:
    """Splits text into chunks by word count with specified overlap."""
    words = text.split()
    if not words:
        return []
    
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size_words, len(words))
        chunk = " ".join(words[start:end])
        if len(chunk.strip()) > 30: # discard trivial stubs
            chunks.append(chunk)
        if end >= len(words):
            break
        start += max(1, chunk_size_words - overlap_words)
    return chunks

_CHUNKS_CACHE = {}

def build_corpus_chunks(chunk_size_words: int = 150, overlap_words: int = 30) -> List[Dict[str, Any]]:
    """
    Ingests all 3 papers and produces chunk records with full citation metadata.
    Uses in-memory cache to avoid redundant PDF parsing.
    """
    cache_key = (chunk_size_words, overlap_words)
    if cache_key in _CHUNKS_CACHE:
        return _CHUNKS_CACHE[cache_key]

    all_chunks = []
    chunk_counter = 0

    for arxiv_id, meta in PAPERS_METADATA.items():
        if not os.path.exists(meta["filename"]):
            raise FileNotFoundError(f"Missing paper: {meta['filename']}. Run download_papers.py first.")
        
        pages = extract_pages(meta["filename"])
        for p in pages:
            page_text = p["text"]
            # Clean up hyphenation at line breaks
            cleaned_text = re.sub(r"(\w+)-\n(\w+)", r"\1\2", page_text)
            cleaned_text = re.sub(r"\s+", " ", cleaned_text).strip()
            
            page_chunks = chunk_text(cleaned_text, chunk_size_words, overlap_words)
            for idx, c_text in enumerate(page_chunks):
                chunk_counter += 1
                all_chunks.append({
                    "chunk_id": f"{meta['short_title']}_p{p['page_num']}_c{idx+1}_{chunk_counter}",
                    "paper_id": arxiv_id,
                    "paper_title": meta["title"],
                    "short_title": meta["short_title"],
                    "page": p["page_num"],
                    "section": p["section"],
                    "text": c_text,
                    "word_count": len(c_text.split()),
                    "approx_tokens": int(len(c_text.split()) * 1.3)
                })

    _CHUNKS_CACHE[cache_key] = all_chunks
    return all_chunks

if __name__ == "__main__":
    print("Ingesting all 3 papers...")
    chunks_small = build_corpus_chunks(chunk_size_words=150, overlap_words=30)
    print(f"Generated {len(chunks_small)} chunks for small setting (~200 tokens each).")
    
    chunks_med = build_corpus_chunks(chunk_size_words=350, overlap_words=50)
    print(f"Generated {len(chunks_med)} chunks for medium setting (~450 tokens each).")
    
    # Preview a chunk
    print("\nSample Chunk:")
    sample = chunks_small[5]
    print(f"ID: {sample['chunk_id']}")
    print(f"Paper: {sample['paper_title']} (arXiv: {sample['paper_id']})")
    print(f"Page: {sample['page']}, Section: {sample['section']}")
    print(f"Text snippet: {sample['text'][:200]}...")
