import os
import time
import httpx

from dotenv import load_dotenv

from langchain_core.prompts import ChatPromptTemplate
from langchain_mistralai import ChatMistralAI

from retriever import hybrid_retriever
from reranker import rerank
from citation_guard import is_supported

load_dotenv()

# -------------------------------------------------------
# LLM
# -------------------------------------------------------

llm = ChatMistralAI(
    model="mistral-large-latest",
    api_key=os.getenv("MISTRAL_API_KEY"),
    temperature=0,
)

# -------------------------------------------------------
# Prompt
# -------------------------------------------------------

prompt = ChatPromptTemplate.from_template(
"""
You are a document-grounded AI assistant.

Answer ONLY using the supplied context.

Rules:

- Never use outside knowledge.
- Never guess.
- Every factual statement must be supported by the supplied context.
- If the answer is not supported by the supplied context, reply EXACTLY:

I don't know based on the provided document.

Context:

{context}

Question:

{question}
"""
)

MAX_RETRIES = 5
RETRY_DELAY = 15


# -------------------------------------------------------
# Main RAG Function
# -------------------------------------------------------

def ask(question: str):

    # ------------------------------------------
    # Hybrid Retrieval
    # ------------------------------------------

    docs = hybrid_retriever.invoke(question)

    # ------------------------------------------
    # Cross Encoder Re-ranking
    # ------------------------------------------

    docs = rerank(
        question=question,
        docs=docs,
        top_k=4,
    )

    # ------------------------------------------
    # Build Context
    # ------------------------------------------

    context = ""

    for doc in docs:

        context += (
            f"[Page {doc.metadata['page'] + 1} | "
            f"Chunk {doc.metadata['chunk']}]\n"
            f"{doc.page_content}\n\n"
        )

    # ------------------------------------------
    # Citation Enforcement
    # ------------------------------------------

    supported = is_supported(
        question,
        context,
    )

    if not supported:

        answer = "I don't know based on the provided document."

    else:

        chain = prompt | llm

        answer = None

        for attempt in range(MAX_RETRIES):

            try:

                response = chain.invoke(
                    {
                        "question": question,
                        "context": context,
                    }
                )

                answer = response.content

                break

            except httpx.HTTPStatusError as e:

                if e.response.status_code == 429:

                    print(
                        f"\nRate limit reached."
                        f"\nWaiting {RETRY_DELAY} seconds..."
                        f"\nRetry {attempt + 1}/{MAX_RETRIES}"
                    )

                    time.sleep(RETRY_DELAY)

                else:
                    raise

            except Exception as e:

                message = str(e)

                if (
                    "429" in message
                    or "rate limit" in message.lower()
                    or "ratelimit" in message.lower()
                ):

                    print(
                        f"\nRate limit detected."
                        f"\nWaiting {RETRY_DELAY} seconds..."
                        f"\nRetry {attempt + 1}/{MAX_RETRIES}"
                    )

                    time.sleep(RETRY_DELAY)

                else:
                    raise

        if answer is None:

            answer = (
                "Unable to generate an answer because the "
                "language model repeatedly exceeded the API rate limit."
            )

    # ------------------------------------------
    # Context List
    # ------------------------------------------

    contexts = [
        doc.page_content
        for doc in docs
    ]

    # ------------------------------------------
    # Citations
    # ------------------------------------------

    citations = []

    for doc in docs:

        citations.append(
            {
                "page": doc.metadata["page"] + 1,
                "chunk": doc.metadata["chunk"],
            }
        )

    return {

        "question": question,

        "answer": answer,

        "contexts": contexts,

        "citations": citations,

        "documents": docs,

    }
    


# -------------------------------------------------------
# Interactive CLI
# -------------------------------------------------------

if __name__ == "__main__":

    print("=" * 80)
    print("Hybrid RAG + Cross Encoder + Citation Enforcement")
    print("=" * 80)

    while True:

        question = input("\nQuestion (type 'exit' to quit): ").strip()

        if question.lower() == "exit":
            break

        if not question:
            continue

        result = ask(question)

        print("\n" + "=" * 80)
        print("ANSWER")
        print("=" * 80)

        print(result["answer"])

        print("\nSources:")

        for citation in result["citations"]:

            print(
                f"- Page {citation['page']} | "
                f"Chunk {citation['chunk']}"
            )

        print("\n" + "=" * 80)
        print("RETRIEVED CHUNKS")
        print("=" * 80)

        for rank, doc in enumerate(result["documents"], start=1):

            print(
                f"\nRank {rank} | "
                f"Page {doc.metadata['page'] + 1} | "
                f"Chunk {doc.metadata['chunk']}"
            )

            print("-" * 80)
            print(doc.page_content)
            print("-" * 80)

    print("\nGoodbye!")