import os
from dotenv import load_dotenv
from src.vectorstore import FaissVectorStore
from src.data_loader import load_all_documents
from src.embedding import EmbeddingPipeline
from langchain_groq import ChatGroq

load_dotenv()

class RAGSearch:
    def __init__(self,
                 persist_dir: str = "faiss_store",
                 embedding_model: str = "all-MiniLM-L6-v2",
                 llm_model: str = "llama-3.3-70b-versatile"):
        # vectorstore: initialized with persist directory only
        self.vectorstore = FaissVectorStore(persist_dir)

        # embedding pipeline used for encoding queries
        self.embedder = EmbeddingPipeline(model_name=embedding_model)

        # check for existing index & metadata (metadata file name used by vectorstore)
        faiss_path = os.path.join(persist_dir, "faiss.index")
        meta_path = os.path.join(persist_dir, "meta.npy")

        if not (os.path.exists(faiss_path) and os.path.exists(meta_path)):
            print("[INFO] FAISS index or metadata missing — building index from documents...")
            # build index from the 'data' directory
            # build_index() will load documents itself using load_all_documents
            self.vectorstore.build_index(docs_dir="data", model_name=embedding_model)
        else:
            self.vectorstore.load()

        groq_api_key = os.getenv("GROQ_API_KEY", "")
        self.llm = ChatGroq(groq_api_key=groq_api_key, model_name=llm_model)
        print(f"[INFO] Groq LLM initialized: {llm_model}")

    def search_and_summarize(self, query: str, top_k: int = 5) -> str:
        # encode query to vector
        qvec = self.embedder.model.encode([query])

        # ensure shape and dtype for FAISS search
        import numpy as np
        qvec = np.asarray(qvec, dtype="float32")
        if qvec.ndim == 1:
            qvec = qvec.reshape(1, -1)

        # run search
        hits = self.vectorstore.search(qvec, top_k=top_k)

        # collect texts from metadata (vectorstore saves text under "_text")
        texts = [h.get("metadata", {}).get("_text", "") for h in hits if h.get("metadata")]
        context = "\n\n".join(texts)
        if not context:
            return "No relevant documents found."

        prompt = f"""Summarize the following context for the query: '{query}'\n\nContext:\n{context}\n\nSummary:"""
        response = self.llm.invoke([prompt])
        # ChatGroq's response object may vary; try to access common fields
        try:
            return getattr(response, "content", None) or getattr(response, "text", None) or str(response)
        except Exception:
            return str(response)


# Example usage
if __name__ == "__main__":
    rag_search = RAGSearch()
