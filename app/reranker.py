from typing import List, Tuple
from sentence_transformers import CrossEncoder
from langchain_core.documents import Document
from app.config import settings

class RerankerService:
    def __init__(self):
        self.model = CrossEncoder(settings.CROSS_ENCODER_MODEL)

    def rerank(self, query: str, documents: List[Document], top_n: int = 3) -> List[Tuple[Document, float]]:
        if not documents:
            return []

        pairs = [[query, doc.page_content] for doc in documents]
        scores = self.model.predict(pairs)

        doc_score_pairs = list(zip(documents, scores))
        doc_score_pairs.sort(key=lambda x: x[1], reverse=True)

        return doc_score_pairs[:top_n]