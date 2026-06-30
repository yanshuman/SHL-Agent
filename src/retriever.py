import json
import os
from typing import List, Dict, Any, Optional
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

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
        self.index: Optional[faiss.Index] = None
        self.model: Optional[SentenceTransformer] = None
        self._load_catalog()
        self._build_or_load_index()

    def _load_catalog(self):
        with open(self.catalog_path, "r", encoding="utf-8") as f:
            self.catalog = json.load(f)
        # Validate required fields
        for item in self.catalog:
            assert "name" in item and "url" in item and "test_type" in item, f"Invalid catalog item: {item}"

    def _build_or_load_index(self):
        if os.path.exists(self.index_path) and os.path.exists(self.index_path + ".mapping"):
            self._load_index()
        else:
            self._build_index()

    def _get_text_for_embedding(self, item: Dict[str, Any]) -> str:
        parts = [
            item["name"],
            item.get("description", ""),
            " ".join(item.get("keywords", [])),
            item.get("test_type", ""),
            " ".join(item.get("job_levels", [])),
        ]
        return " ".join(parts)

    def _build_index(self):
        print("Building FAISS index...")
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        texts = [self._get_text_for_embedding(item) for item in self.catalog]
        embeddings = self.model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        faiss.normalize_L2(embeddings)
        self.index.add(embeddings)
        faiss.write_index(self.index, self.index_path)
        mapping = [item["name"] for item in self.catalog]
        with open(self.index_path + ".mapping", "w", encoding="utf-8") as f:
            json.dump(mapping, f)
        print(f"Index built with {len(self.catalog)} items.")

    def _load_index(self):
        print("Loading FAISS index...")
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.index = faiss.read_index(self.index_path)
        with open(self.index_path + ".mapping", "r", encoding="utf-8") as f:
            mapping = json.load(f)
        # Reorder catalog to match mapping if needed
        name_to_item = {item["name"]: item for item in self.catalog}
        self.catalog = [name_to_item[name] for name in mapping]
        print(f"Index loaded with {len(self.catalog)} items.")

    def search(self, query: str, top_k: int = 10, filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        if self.model is None or self.index is None:
            raise RuntimeError("Index not initialized")
        
        query_embedding = self.model.encode([query], convert_to_numpy=True)
        faiss.normalize_L2(query_embedding)
        scores, indices = self.index.search(query_embedding, top_k * 3)
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            item = self.catalog[idx].copy()
            item["_score"] = float(score)
            
            # Apply filters
            if filters:
                skip = False
                if "test_types" in filters and item["test_type"] not in filters["test_types"]:
                    skip = True
                if "job_levels" in filters and not any(level in item.get("job_levels", []) for level in filters["job_levels"]):
                    skip = True
                if "keywords" in filters:
                    item_text = self._get_text_for_embedding(item).lower()
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
