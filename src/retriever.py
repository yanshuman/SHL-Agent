import json
import os
from typing import List, Dict, Any, Optional
import numpy as np

class CatalogRetriever:
    def __init__(self, catalog_path: str = None, index_path: str = None):
        if catalog_path is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            catalog_path = os.path.join(base_dir, "data", "catalog.json")
        if index_path is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            index_path = os.path.join(base_dir, "data", "catalog.index")
        self.catalog_path = catalog_path
        self.index_path = index_path
        self.catalog: List[Dict[str, Any]] = []
        self.vectors = None
        self.vocab = {}
        self.idf = None
        self._load_catalog()
        self._build_or_load_index()

    def _load_catalog(self):
        with open(self.catalog_path, "r", encoding="utf-8") as f:
            self.catalog = json.load(f)
        for item in self.catalog:
            assert "name" in item and "url" in item and "test_type" in item

    def _get_text(self, item: Dict[str, Any]) -> str:
        parts = [
            item["name"],
            item.get("description", ""),
            " ".join(item.get("keywords", [])),
            item.get("test_type", ""),
            " ".join(item.get("job_levels", [])),
        ]
        return " ".join(parts).lower()

    def _tokenize(self, text: str) -> List[str]:
        return text.lower().replace(",", " ").replace(".", " ").split()

    def _build_tfidf(self, texts: List[str]):
        # Build vocabulary
        vocab = {}
        doc_freq = {}
        for text in texts:
            tokens = set(self._tokenize(text))
            for token in tokens:
                if token not in vocab:
                    vocab[token] = len(vocab)
                doc_freq[token] = doc_freq.get(token, 0) + 1
        
        N = len(texts)
        idf = np.zeros(len(vocab))
        for token, idx in vocab.items():
            idf[idx] = np.log((N + 1) / (doc_freq[token] + 1)) + 1
        
        # Build document vectors
        vectors = np.zeros((N, len(vocab)))
        for i, text in enumerate(texts):
            tokens = self._tokenize(text)
            for token in tokens:
                if token in vocab:
                    vectors[i, vocab[token]] += 1
            # TF-IDF weighting
            vectors[i] *= idf
        
        # L2 normalize
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1
        vectors = vectors / norms
        
        return vectors, vocab, idf

    def _build_or_load_index(self):
        import pickle
        tfidf_path = self.index_path + ".tfidf"
        if os.path.exists(tfidf_path):
            print("Loading lightweight TF-IDF index...")
            with open(tfidf_path, "rb") as f:
                data = pickle.load(f)
            self.vectors = data["vectors"]
            self.vocab = data["vocab"]
            self.idf = data["idf"]
            print(f"Index loaded with {len(self.catalog)} items, vocab size {len(self.vocab)}.")
        else:
            print("Building lightweight TF-IDF index...")
            texts = [self._get_text(item) for item in self.catalog]
            self.vectors, self.vocab, self.idf = self._build_tfidf(texts)
            with open(tfidf_path, "wb") as f:
                pickle.dump({"vectors": self.vectors, "vocab": self.vocab, "idf": self.idf}, f)
            print(f"Index built with {len(self.catalog)} items, vocab size {len(self.vocab)}.")

    def _encode_query(self, query: str) -> np.ndarray:
        tokens = self._tokenize(query)
        vec = np.zeros(len(self.vocab))
        for token in tokens:
            if token in self.vocab:
                vec[self.vocab[token]] += 1
        vec *= self.idf
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec.reshape(1, -1)

    def search(self, query: str, top_k: int = 10, filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        if self.vectors is None:
            raise RuntimeError("Index not initialized")
        
        query_vec = self._encode_query(query)
        # Cosine similarity via dot product (vectors are L2 normalized)
        scores = (self.vectors @ query_vec.T).flatten()
        # Get top indices
        top_indices = np.argsort(scores)[::-1]
        
        results = []
        for idx in top_indices:
            if scores[idx] <= 0:
                continue
            item = self.catalog[int(idx)].copy()
            item["_score"] = float(scores[idx])
            
            # Apply filters
            if filters:
                skip = False
                if "test_types" in filters and item["test_type"] not in filters["test_types"]:
                    skip = True
                if "job_levels" in filters and not any(level in item.get("job_levels", []) for level in filters["job_levels"]):
                    skip = True
                if "keywords" in filters:
                    item_text = self._get_text(item).lower()
                    if not any(kw.lower() in item_text for kw in filters["keywords"]):
                        skip = True
                if skip:
                    continue
            
            results.append(item)
            if len(results) >= top_k:
                break
        
        return results

    def get_by_name(self, name: str) -> Optional[Dict[str, Any]]:
        for item in self.catalog:
            if item["name"].lower() == name.lower():
                return item.copy()
        return None

    def get_all_names(self) -> List[str]:
        return [item["name"] for item in self.catalog]

    def compare(self, name1: str, name2: str) -> Optional[Dict[str, Any]]:
        item1 = self.get_by_name(name1)
        item2 = self.get_by_name(name2)
        if not item1 or not item2:
            return None
        return {
            "item1": item1,
            "item2": item2,
            "differences": {
                "test_type": (item1["test_type"], item2["test_type"]),
                "duration": (item1.get("duration", ""), item2.get("duration", "")),
                "description": (item1.get("description", ""), item2.get("description", "")),
            }
        }

# Singleton retriever instance
_retriever: Optional[CatalogRetriever] = None

def get_retriever() -> CatalogRetriever:
    global _retriever
    if _retriever is None:
        _retriever = CatalogRetriever()
    return _retriever
