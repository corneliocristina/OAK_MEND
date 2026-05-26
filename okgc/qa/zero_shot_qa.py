import os

from okgc.utils.filesystem import load_prompt
from okgc.utils.openai_wrapper import OpenAIClient
from okgc.utils.usage import UsageInfo


class ZeroShotQA:
    def __init__(
        self,
        client: OpenAIClient,
        *,
        prompts_path: str | None = None,
        verbose: bool = False,
    ):
        if prompts_path is None:
            prompts_path = os.path.join("prompts", "zero-shot-qa")
        self.client = client
        self.verbose = verbose
        self.usage_info = UsageInfo()

        self.prompts: dict[str, str] = {
            "qa": load_prompt(os.path.join(prompts_path, "qa_prompt.txt")),
        }

    def ask(self, question: str) -> str:
        user_prompt = f"Question:\n{question}\n\n"
        answer, usage = self.client.get_completion(
            self.prompts["qa"], user_prompt, transform_to_json=False
        )
        self.usage_info += usage

        return answer
