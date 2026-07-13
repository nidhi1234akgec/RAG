import os

from dotenv import load_dotenv

from deepeval.metrics import FaithfulnessMetric
from deepeval.models import DeepEvalBaseLLM

from langchain_mistralai import ChatMistralAI

load_dotenv()


class MistralDeepEvalLLM(DeepEvalBaseLLM):
    """
    Wrapper so DeepEval can use Mistral as the judge model.
    """

    def __init__(self):

        self.model = ChatMistralAI(
            model="mistral-large-latest",
            api_key=os.getenv("MISTRAL_API_KEY"),
            temperature=0,
        )

    def load_model(self):
        return self.model

    def generate(self, prompt: str) -> str:

        response = self.model.invoke(prompt)

        return response.content

    async def a_generate(self, prompt: str) -> str:

        response = await self.model.ainvoke(prompt)

        return response.content

    def get_model_name(self):

        return "mistral-large-latest"


judge_llm = MistralDeepEvalLLM()


faithfulness_metric = FaithfulnessMetric(

    threshold=0.85,

    model=judge_llm,

    include_reason=True,

    verbose_mode=True,

)