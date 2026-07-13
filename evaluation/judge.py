import json

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama


# -------------------------------------------------------
# Local LLM Judge
# -------------------------------------------------------

judge_llm = ChatOllama(
    model="qwen2.5:7b",
    temperature=0,
)

# -------------------------------------------------------
# Judge Prompt
# -------------------------------------------------------

judge_prompt = ChatPromptTemplate.from_template(
"""
You are an expert evaluator for Retrieval-Augmented Generation (RAG) systems.

Your job is to determine whether the generated answer is fully supported by the retrieved context.

You must NEVER use outside knowledge.

You must ONLY consider the retrieved context.

----------------------------------------

Question:

{question}

----------------------------------------

Retrieved Context:

{context}

----------------------------------------

Generated Answer:

{answer}

----------------------------------------

Evaluate the answer.

Return ONLY valid JSON.

{{
    "faithfulness": true,
    "score": 1.0,
    "reason": "Every factual claim is directly supported by the retrieved context."
}}

Scoring Rules

1.0
Every claim is supported.

0.75
Minor unsupported wording.

0.50
Some unsupported claims.

0.25
Mostly unsupported.

0.0
Hallucinated answer.

Output JSON only.
"""
)

parser = StrOutputParser()

chain = judge_prompt | judge_llm | parser


# -------------------------------------------------------
# Judge Function
# -------------------------------------------------------

def judge_answer(
    question: str,
    answer: str,
    contexts: list[str],
):

    context = "\n\n".join(contexts)

    response = chain.invoke(
        {
            "question": question,
            "answer": answer,
            "context": context,
        }
    )

    # ---------------------------------------------------
    # Parse JSON
    # ---------------------------------------------------

    try:

        result = json.loads(response)

    except Exception:

        result = {
            "faithfulness": False,
            "score": 0.0,
            "reason": "Judge returned invalid JSON.",
        }

    # ---------------------------------------------------

    result.setdefault("faithfulness", False)
    result.setdefault("score", 0.0)
    result.setdefault("reason", "")

    return result