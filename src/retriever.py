"""
Information Retrieval Engine for 'The Relay'.
Supports configurable chunk sizes, candidate count k, and optional re-ranking.
Produces structured citations with exact page numbers and section titles.
"""

import math
import re
from typing import List, Dict, Any, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

from src.ingestion import build_corpus_chunks

class BM25Scorer:
    """Lightweight, self-contained BM25 Okapi implementation for precise academic retrieval."""
    def __init__(self, corpus: List[str], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.corpus_size = len(corpus)
        self.doc_lens = [len(doc.lower().split()) for doc in corpus]
        self.avg_dl = sum(self.doc_lens) / max(1, self.corpus_size)
        self.tokenized_corpus = [self._tokenize(doc) for doc in corpus]
        
        # Calculate Document Frequencies (DF)
        self.df = {}
        for doc in self.tokenized_corpus:
            unique_terms = set(doc)
            for term in unique_terms:
                self.df[term] = self.df.get(term, 0) + 1
        
        # Calculate IDF
        self.idf = {}
        for term, freq in self.df.items():
            self.idf[term] = math.log((self.corpus_size - freq + 0.5) / (freq + 0.5) + 1.0)

    def _tokenize(self, text: str) -> List[str]:
        return re.findall(r"\b[a-zA-Z0-9_\-\.]{2,}\b", text.lower())

    def score(self, query: str) -> np.ndarray:
        q_tokens = self._tokenize(query)
        scores = np.zeros(self.corpus_size)
        
        for q in q_tokens:
            if q not in self.idf:
                continue
            q_idf = self.idf[q]
            for doc_idx, doc_tokens in enumerate(self.tokenized_corpus):
                tf = doc_tokens.count(q)
                if tf > 0:
                    doc_len = self.doc_lens[doc_idx]
                    numerator = tf * (self.k1 + 1)
                    denominator = tf + self.k1 * (1 - self.b + self.b * (doc_len / self.avg_dl))
                    scores[doc_idx] += q_idf * (numerator / denominator)
        return scores

class PaperRetriever:
    def __init__(self, chunk_size_words: int = 150, overlap_words: int = 30):
        self.chunk_size_words = chunk_size_words
        self.overlap_words = overlap_words
        self.chunks = build_corpus_chunks(chunk_size_words, overlap_words)
        
        self.corpus_texts = [c["text"] for c in self.chunks]
        self.bm25 = BM25Scorer(self.corpus_texts)
        
        # Secondary TF-IDF index for n-gram matching
        self.tfidf = TfidfVectorizer(ngram_range=(1, 2), stop_words='english')
        self.tfidf_matrix = self.tfidf.fit_transform(self.corpus_texts)

    def retrieve(self, query: str, k: int = 5, rerank: bool = False, top_n_rerank: int = 4) -> List[Dict[str, Any]]:
        """
        Retrieves top-k chunks. If rerank=True, retrieves k initial candidates
        and re-ranks down to top_n_rerank based on hybrid score and exact lexical matches.
        """
        # Primary BM25 score
        bm25_scores = self.bm25.score(query)
        
        # Secondary TF-IDF cosine similarity
        q_vec = self.tfidf.transform([query])
        tfidf_scores = (self.tfidf_matrix * q_vec.T).toarray().flatten()
        
        # Hybrid score (normalized blend)
        max_bm25 = max(1e-5, np.max(bm25_scores))
        max_tfidf = max(1e-5, np.max(tfidf_scores))
        hybrid_scores = 0.65 * (bm25_scores / max_bm25) + 0.35 * (tfidf_scores / max_tfidf)
        
        # Initial ranking
        initial_indices = np.argsort(hybrid_scores)[::-1][:k]
        candidates = []
        for idx in initial_indices:
            c = dict(self.chunks[idx])
            c["score"] = float(hybrid_scores[idx])
            candidates.append(c)

        if not rerank:
            return candidates

        # Re-ranking logic:
        # Boost candidates with exact phrase matches, section/title relevance, and numerical precision
        q_terms = [t.lower() for t in re.findall(r"\b[a-zA-Z0-9_\-\.]{2,}\b", query)]
        reranked = []
        for c in candidates:
            score = c["score"]
            text_lower = c["text"].lower()
            
            # Exact query phrase boost
            if query.lower() in text_lower:
                score += 0.4
            
            # Key term density in section title
            section_lower = c["section"].lower()
            for term in q_terms:
                if term in section_lower:
                    score += 0.15
                if term.isdigit() and term in text_lower: # numbers matter for hyperparams!
                    score += 0.1
            
            c["rerank_score"] = score
            reranked.append(c)
            
        reranked.sort(key=lambda x: x["rerank_score"], reverse=True)
        return reranked[:top_n_rerank]

    def format_citations_prompt(self, passages: List[Dict[str, Any]]) -> str:
        """Formats passages into a clean prompt block with citation anchors."""
        formatted_blocks = []
        for idx, p in enumerate(passages, 1):
            block = (
                f"--- Passage {idx} ---\n"
                f"Paper: {p['paper_title']} (arXiv: {p['paper_id']})\n"
                f"Location: Page {p['page']}, Section: {p['section']}\n"
                f"Content:\n{p['text']}\n"
            )
            formatted_blocks.append(block)
        return "\n".join(formatted_blocks)

# Pre-instantiate standard retrieval configs for quick switching
RETRIEVAL_CONFIGS = {
    "setting_1": {"chunk_size_words": 150, "overlap_words": 30, "k": 3, "rerank": False, "desc": "Small chunks (200t), k=3, no rerank"},
    "setting_2": {"chunk_size_words": 350, "overlap_words": 50, "k": 5, "rerank": False, "desc": "Medium chunks (450t), k=5, no rerank"},
    "setting_3": {"chunk_size_words": 350, "overlap_words": 50, "k": 8, "rerank": True, "top_n": 4, "desc": "Medium chunks (450t), k=8 -> rerank to top 4"}
}

def get_retriever_for_setting(setting_name: str) -> Tuple[PaperRetriever, Dict[str, Any]]:
    config = RETRIEVAL_CONFIGS.get(setting_name, RETRIEVAL_CONFIGS["setting_2"])
    retriever = PaperRetriever(config["chunk_size_words"], config["overlap_words"])
    return retriever, config

if __name__ == "__main__":
    print("Testing Retriever on a sample technical query...")
    retriever, cfg = get_retriever_for_setting("setting_3")
    query = "What is the frame rate and bitrate of the Mimi audio codec in Moshi?"
    results = retriever.retrieve(query, k=cfg["k"], rerank=cfg["rerank"], top_n_rerank=cfg.get("top_n", 4))
    
    print(f"\nRetrieved {len(results)} passages for: '{query}'")
    for r in results:
        print(f"-> [{r['short_title']} p.{r['page']} § {r['section']}] score={r.get('rerank_score', r['score']):.3f}")
        print(f"   Excerpt: {r['text'][:120]}...\n")

