import os
import json
from typing import List, Dict, Any
import langchain
from langchain_community.cache import RedisSemanticCache
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document

from app.config import settings
from app.reranker import RerankerService
from app.ingestion import hierarchical_chunking, add_to_chroma

class AdvancedRAGPipeline:
    def __init__(self):
        os.environ["GOOGLE_API_KEY"] = settings.GEMINI_API_KEY
        self.embeddings = GoogleGenerativeAIEmbeddings(model=settings.EMBEDDING_MODEL)
        self.llm = ChatGoogleGenerativeAI(model=settings.LLM_MODEL, temperature=0.1)
        self.reranker = RerankerService()
        
        # Redis Semantic Cache
        langchain.llm_cache = RedisSemanticCache(
            redis_url=settings.REDIS_URL,
            embedding=self.embeddings
        )

        self.vector_store = Chroma(
            collection_name="nexus_postmortems",
            embedding_function=self.embeddings,
            persist_directory=settings.CHROMA_PERSIST_DIR
        )
        self._sync_bm25()

    def _sync_bm25(self):
        existing_docs = self.vector_store.get()
        if existing_docs and existing_docs.get("documents"):
            docs = [
                Document(page_content=content, metadata=existing_docs["metadatas"][i] or {})
                for i, content in enumerate(existing_docs["documents"])
            ]
            self.bm25_retriever = BM25Retriever.from_documents(docs)
            self.bm25_retriever.k = settings.TOP_K_RETRIEVAL
        else:
            self.bm25_retriever = None

    def ingest(self, file_path: str) -> int:
        chunks = hierarchical_chunking(file_path)
        num_added = add_to_chroma(chunks, self.vector_store)
        if num_added > 0:
            self._sync_bm25()
        return num_added

    def _get_text(self, response) -> str:
        """Extrait le texte brut, que la réponse de l'API soit une string ou une liste de blocs."""
        content = response.content
        if isinstance(content, str):
            return content
        elif isinstance(content, list) and len(content) > 0:
            if isinstance(content[0], dict) and "text" in content[0]:
                return content[0]["text"]
            return str(content[0])
        return str(content)

    def query_expansion(self, original_query: str) -> List[str]:
        prompt = f"""
        Tu es un expert technique. Reformule la question de l'utilisateur en 2 requêtes de recherche distinctes 
        (synonymes, mots-clés techniques différents) pour chercher dans des post-mortems d'ingénierie.
        Ne donne pas la réponse. Renvoie uniquement les requêtes séparées par des sauts de ligne.
        Question: {original_query}
        """
        response = self.llm.invoke(prompt)
        text_content = self._get_text(response)
        expanded = [q.strip() for q in text_content.split('\n') if q.strip()]
        return [original_query] + expanded[:2]

    def extract_query_filters(self, query: str) -> dict:
        prompt = f"""
        Extrait le nom de l'entreprise mentionnée dans cette question. 
        Renvoie uniquement un JSON {{"company": "nom_entreprise"}}. Si aucune, renvoie {{"company": null}}.
        Question : {query}
        """
        try:
            response = self.llm.invoke(prompt)
            text_content = self._get_text(response)
            content = text_content.replace('```json', '').replace('```', '').strip()
            filters = json.loads(content)
            if filters.get("company"):
                return {"company": filters["company"].lower()}
        except:
            pass
        return {}

    def reciprocal_rank_fusion(self, list_of_results: List[List[Document]], k=60) -> List[Document]:
        rrf_map = {}
        doc_map = {}
        for doc_list in list_of_results:
            for rank, doc in enumerate(doc_list, 1):
                doc_id = doc.metadata.get("id", doc.page_content[:50])
                if doc_id not in rrf_map:
                    rrf_map[doc_id] = 0.0
                    doc_map[doc_id] = doc
                rrf_map[doc_id] += 1.0 / (k + rank)
                
        sorted_ids = sorted(rrf_map.keys(), key=lambda x: rrf_map[x], reverse=True)
        return [doc_map[doc_id] for doc_id in sorted_ids]

    def hybrid_search(self, query: str, filters: dict) -> List[Document]:
        queries = self.query_expansion(query)
        all_dense = []
        all_sparse = []
        
        for q in queries:
            dense = self.vector_store.similarity_search(
                q, k=settings.TOP_K_RETRIEVAL, filter=filters if filters else None
            )
            all_dense.append(dense)
            
            if self.bm25_retriever:
                sparse = self.bm25_retriever.invoke(q)
                if filters and "company" in filters:
                    sparse = [d for d in sparse if d.metadata.get("company") == filters["company"]]
                all_sparse.append(sparse)

        flattened_results = all_dense + all_sparse
        fused_docs = self.reciprocal_rank_fusion(flattened_results)
        return fused_docs[:10]

    def query(self, user_query: str) -> Dict[str, Any]:
        filters = self.extract_query_filters(user_query)
        candidate_docs = self.hybrid_search(user_query, filters)

        ranked_pairs = self.reranker.rerank(user_query, candidate_docs, top_n=settings.TOP_K_RERANK)
        top_docs = [pair[0] for pair in ranked_pairs]

        if not top_docs:
            return {"answer": "Aucun document pertinent trouvé.", "sources": []}

        context = "\n\n".join([f"Source: {d.metadata.get('id')}\n{d.page_content}" for d in top_docs])
        prompt = f"""
        Tu es un assistant expert en post-mortems d'ingénierie.
        Réponds à la question en utilisant UNIQUEMENT le contexte fourni.
        
        Contexte :
        {context}
        
        Question : {user_query}
        Réponse :
        """
        
        response = self.llm.invoke(prompt)
        text_content = self._get_text(response)

        return {
            "query": user_query,
            "filters_applied": filters,
            "answer": text_content,
            "sources": [{"id": d.metadata.get("id"),"text": d.page_content, "score": float(score)} for d, score in ranked_pairs]
        }