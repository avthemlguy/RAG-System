import os
import time
import json
import numpy as np
import faiss
from typing import Optional, List, Dict, Any

from src.embedding import EmbeddingPipeline
from src.data_loader import load_all_documents


class FaissVectorStore:
    def __init__(self, store_dir: str = "faiss_store", index_name: str = "faiss.index", meta_name: str = "meta.npy"):
        self.store_dir = store_dir
        self.faiss_path = os.path.join(store_dir, index_name)
        self.meta_path = os.path.join(store_dir, meta_name)
        self.index: Optional[faiss.Index] = None

    def load(self, auto_build: bool = True) -> None:
        """Load FAISS index from disk.

        If the index file is missing and `auto_build` is True, this will attempt to
        build a new index from documents returned by `load_all_documents("data")`.
        If the index exists but is corrupted, it will be backed up and rebuilt.
        """
        os.makedirs(self.store_dir, exist_ok=True)

        # If index file missing -> optionally build it automatically
        if not os.path.exists(self.faiss_path):
            if auto_build:
                print(f"[WARN] FAISS index not found at {self.faiss_path}. Building a new one now...")
                self.build_index()
                return
            else:
                raise FileNotFoundError(f"FAISS index not found at {self.faiss_path}. Run build_index() to create it.")

        # Try to read existing index; if corrupt -> backup and rebuild
        try:
            self.index = faiss.read_index(self.faiss_path)
            print(f"[INFO] Loaded FAISS index from {self.faiss_path}")
            return
        except RuntimeError as e:
            ts = time.strftime("%Y%m%d-%H%M%S")
            corrupt_backup = f"{self.faiss_path}.corrupt.{ts}"
            try:
                os.rename(self.faiss_path, corrupt_backup)
                print(f"[WARN] FAISS read failed: backed up corrupted index to {corrupt_backup}")
            except Exception as rename_err:
                print(f"[WARN] Failed to backup corrupted index: {rename_err} (orig error: {e})")
            print("[INFO] Rebuilding FAISS index from documents...")
            self.build_index()
        except Exception as e:
            raise RuntimeError(f"Failed to read FAISS index: {e}") from e

    def build_index(self,
                    docs_dir: str = "data",
                    model_name: Optional[str] = None,
                    chunk_size: int = 1000,
                    chunk_overlap: int = 200,
                    index_factory: Optional[str] = None) -> None:
        """Build and save a FAISS index from documents.

        - `index_factory` (optional): if provided and valid, used to create more advanced indices
          via `faiss.index_factory`. If None uses a simple IndexFlatL2 which requires no training.
        """
        # load documents
        docs = load_all_documents(docs_dir)
        if not docs:
            raise RuntimeError(f"No documents found in {docs_dir} - cannot build FAISS index.")

        # instantiate embedding pipeline
        emb_pipe = EmbeddingPipeline(model_name=model_name or "all-MiniLM-L6-v2",
                                     chunk_size=chunk_size, chunk_overlap=chunk_overlap)

        chunks = emb_pipe.chunk_documents(docs)
        if not chunks:
            raise RuntimeError("Chunking returned no chunks - check your documents and splitter settings.")

        embeddings = emb_pipe.embed_chunks(chunks)
        emb = np.asarray(embeddings)
        if emb.ndim != 2:
            raise RuntimeError(f"Embeddings must be 2-dimensional array (N,D). Got shape: {emb.shape}")
        emb = emb.astype("float32")

        d = emb.shape[1]

        # Create index: use a simple IndexFlatL2 by default (no training required)
        if index_factory:
            try:
                index = faiss.index_factory(d, index_factory)
                # If the index needs training, train it on the vectors
                if index.is_trained is False:
                    print("[INFO] Training FAISS index (index reports is_trained=False)...")
                    index.train(emb)
            except Exception as e:
                print(f"[WARN] Failed to create advanced index via factory '{index_factory}': {e}. Falling back to IndexFlatL2.")
                index = faiss.IndexFlatL2(d)
        else:
            index = faiss.IndexFlatL2(d)

        index.add(emb)

        # ensure store dir exists
        os.makedirs(self.store_dir, exist_ok=True)

        # write index and validate by reading it back
        try:
            faiss.write_index(index, self.faiss_path)
        except Exception as e:
            raise RuntimeError(f"Failed to write FAISS index to {self.faiss_path}: {e}") from e

        # quick read-back validation
        try:
            test_index = faiss.read_index(self.faiss_path)
            # simple sanity check
            if test_index.ntotal != index.ntotal:
                raise RuntimeError("Validation read-back mismatch: vector counts differ")
        except Exception as e:
            raise RuntimeError(f"Validation failed reading back FAISS index: {e}") from e

        # save metadata (make serializable), mapping vector -> text + metadata
        metadata: List[Dict[str, Any]] = []
        for ch in chunks:
            meta: Dict[str, Any] = {}
            if hasattr(ch, "metadata") and isinstance(ch.metadata, dict):
                # shallow copy; non-serializable values are stringified
                for k, v in ch.metadata.items():
                    try:
                        json.dumps(v)
                        meta[k] = v
                    except (TypeError, ValueError):
                        meta[k] = str(v)
            # text content
            meta["_text"] = getattr(ch, "page_content", str(ch))
            metadata.append(meta)

        # Save metadata as numpy object array for simplicity
        try:
            np.save(self.meta_path, np.array(metadata, dtype=object), allow_pickle=True)
        except Exception as e:
            print(f"[WARN] Failed to save metadata to {self.meta_path}: {e}")

        print(f"[INFO] Built and wrote FAISS index to {self.faiss_path}; saved metadata to {self.meta_path}")

        # load into memory
        self.index = index

    def search(self, query_vector: np.ndarray, top_k: int = 5) -> List[Dict[str, Any]]:
        """Run a similarity search against the loaded index.

        Returns a list of dicts with keys: score, index, metadata (if available)
        """
        if self.index is None:
            raise RuntimeError("FAISS index not loaded. Call load() first.")

        q = np.asarray(query_vector, dtype="float32")
        if q.ndim == 1:
            q = q.reshape(1, -1)
        if q.shape[1] != self.index.d:
            raise ValueError(f"Query vector dim {q.shape[1]} does not match index dim {self.index.d}")

        distances, indices = self.index.search(q, top_k)
        hits = []
        # load metadata if exists
        meta_list = None
        if os.path.exists(self.meta_path):
            try:
                meta_list = np.load(self.meta_path, allow_pickle=True).tolist()
            except Exception:
                meta_list = None

        for dist, idx in zip(distances[0], indices[0]):
            if idx < 0:
                continue
            item = {"score": float(dist), "index": int(idx)}
            if meta_list and idx < len(meta_list):
                item["metadata"] = meta_list[idx]
            hits.append(item)
        return hits


# simple helper to rebuild index from script
if __name__ == "__main__":
    store = FaissVectorStore()
    store.build_index()
