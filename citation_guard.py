import os

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_mistralai import ChatMistralAI

load_dotenv()

llm = ChatMistralAI(
    model="mistral-large-latest",
    api_key=os.getenv("MISTRAL_API_KEY"),
    temperature=0,
)

verification_prompt = ChatPromptTemplate.from_template(
    """
You are a retrieval verification system.

Your task is to determine whether the supplied context contains enough
information to answer the user's question.

Rules:

- Use ONLY the provided context.
- If any important part of the answer is missing, return UNSUPPORTED.
- Do NOT use outside knowledge.
- Do NOT guess.
- Do NOT infer facts not explicitly supported.

Context:
{context}

Question:
{question}

Return ONLY one word.

SUPPORTED

or

UNSUPPORTED
"""
)


def is_supported(question: str, context: str) -> bool:

    chain = verification_prompt | llm

    response = chain.invoke(
        {
            "question": question,
            "context": context,
        }
    )

    return response.content.strip().upper() == "SUPPORTED"