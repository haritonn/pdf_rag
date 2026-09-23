from collections.abc import Sequence

from fastembed.rerank.cross_encoder import TextCrossEncoder
from langchain_core.documents import Document

from .base import Reranker

class FastEmbedReranker:
    def __init__(
        self,
        model_name: str = "Xenova/ms-marco-MiniLM-L-6-v2",
        providers: Sequence[str] | None = None,
    ) -> None:
        self.model = TextCrossEncoder(
            model_name=model_name,
            providers=list(providers or ["CPUExecutionProvider"]),
        )

    def rerank(
        self,
        query: str,
        documents: Sequence[Document],
        top_k: int,
    ) -> list[Document]:
        if not documents:
            return []

        scores = list(
            self.model.rerank(
                query,
                [document.page_content for document in documents],
            )
        )
        ranked = sorted(
            zip(scores, documents, strict=True),
            key=lambda item: float(item[0]),
            reverse=True,
        )

        result = []
        for score, document in ranked[:top_k]:
            result.append(
                Document(
                    page_content=document.page_content,
                    metadata={
                        **document.metadata,
                        "_rerank_score": float(score),
                    },
                )
            )
        return result
