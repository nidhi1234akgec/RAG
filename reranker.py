from sentence_transformers import CrossEncoder

# Load once
reranker = CrossEncoder(
    "BAAI/bge-reranker-v2-m3"
)


def rerank(question, docs, top_k=4):
    """
    Rerank retrieved documents using a cross-encoder.

    Args:
        question (str)
        docs (List[Document])
        top_k (int)

    Returns:
        List[Document]
    """

    pairs = [
        (question, doc.page_content)
        for doc in docs
    ]

    scores = reranker.predict(pairs)

    ranked = sorted(
        zip(scores, docs),
        key=lambda x: x[0],
        reverse=True,
    )

    return [
        doc
        for score, doc in ranked[:top_k]
    ]