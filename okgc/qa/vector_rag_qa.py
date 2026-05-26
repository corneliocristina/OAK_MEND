import os

from okgc.utils.filesystem import load_prompt
from okgc.utils.openai_wrapper import OpenAIClient
from okgc.utils.sent_embed import SentenceEmbedder
from okgc.utils.usage import UsageInfo
from okgc.utils.vector_index import StrIndex


class VectorRAGQA:
    def __init__(
        self,
        client: OpenAIClient,
        sent_embed: SentenceEmbedder,
        index: StrIndex,
        *,
        prompts_path: str | None = None,
        verbose: bool = False,
    ):
        if prompts_path is None:
            prompts_path = os.path.join("prompts", "vector-rag")
        self.client = client
        self.sent_embed = sent_embed
        self.index = index
        self.verbose = verbose
        self.usage_info = UsageInfo()

        self.prompts: dict[str, str] = {
            "qa": load_prompt(os.path.join(prompts_path, "qa_prompt.txt")),
        }

    def ask(self, question: str, *, max_context_docs: int = 15) -> str:
        # Retrieve the top-k relevant documents from the index
        search_output = self.index.search(question, k=max_context_docs)
        assert isinstance(search_output, list) and len(search_output) == 1
        relevant_documents = search_output[0]

        # Ask the LLM to answer to the question based on the relevant documents
        user_prompt = f"Question:\n{question}\n\n"
        user_prompt += f"Relevant documents:\n"
        user_prompt += "\n".join(f"- {doc}" for doc in relevant_documents)
        answer, usage = self.client.get_completion(
            self.prompts["qa"], user_prompt, transform_to_json=False
        )
        self.usage_info += usage

        return answer
