import json
import re

import json_repair
import openai
from tenacity import retry, stop_after_attempt, wait_random_exponential

from okgc.utils.usage import UsageInfo

PATTERNS = {
    "json_blocks": re.compile(r"```json\s*(\{.*?\}|\[.*?\])\s*```", flags=re.DOTALL),
    "json_inline": re.compile(r"(\{.*?\}|\[.*?\])", flags=re.DOTALL),
    "reasoning_pattern": re.compile(r".*</think>", flags=re.DOTALL),
}


class OpenAIClient:
    def __init__(
        self,
        model: str,
        base_url: str,
        *,
        api_key: str | None = None,
        seed: int = 42,
        verbose: bool = False,
    ):
        api_key = "" if api_key is None else api_key
        self.client = openai.OpenAI(base_url=base_url, api_key=api_key)
        if not model:
            model = self.client.models.list().data[0].id
        self.model = model
        self.seed = seed
        self.verbose = verbose
        #
        self._avoid_system_prompt = any(m in model for m in ["DeepSeek-R1"])

    def extract_json(self, text: str) -> dict | list | str:
        try:
            return json.loads(text)  # type: ignore
        except:
            try:
                return json_repair.loads(text)  # type: ignore
            except:
                for name in ["json_blocks", "json_inline"]:
                    pattern = PATTERNS[name]
                    match = re.search(pattern, text)
                    if match:
                        return json_repair.loads(match.group(1))  # type: ignore
        raise ValueError(f"JSON extraction failed from text content: {text}")

    @retry(
        wait=wait_random_exponential(multiplier=1, max=60),
        stop=stop_after_attempt(5),
    )
    def get_completion(
        self, system_prompt: str, user_prompt: str, transform_to_json: bool = True
    ) -> tuple[dict | list | str, UsageInfo]:
        if self._avoid_system_prompt:
            messages = [
                {"role": "user", "content": f"{system_prompt}\n\n{user_prompt}"},
            ]
        else:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            seed=self.seed,
        )
        assert response.usage is not None
        usage_info = UsageInfo(
            response.usage.prompt_tokens, response.usage.completion_tokens
        )
        assert len(response.choices) == 1
        if response.choices[0].message.content is None:
            response.choices[0].message.content = ""
        assert response.choices[0].message.content is not None
        content = response.choices[0].message.content.strip()
        if (
            hasattr(response.choices[0].message, "reasoning_content")
            and response.choices[0].message.reasoning_content is not None
        ):
            reasoning_content = response.choices[0].message.reasoning_content.strip()
        else:
            reasoning_content = None

        if self.verbose:
            print("System prompt:")
            print(system_prompt)
            print()
            print("User prompt:")
            print(user_prompt)
            print()
            if reasoning_content:
                print("LLM response reasoning content:")
                print(reasoning_content)
                print()
            print("LLM response content:")
            print(content)
            print()

        content = re.sub(PATTERNS["reasoning_pattern"], r"", content)
        output = self.extract_json(content) if transform_to_json else content
        return output, usage_info

    @retry(
        wait=wait_random_exponential(multiplier=1, max=60),
        stop=stop_after_attempt(5),
    )
    def get_embeddings(
        self, text: str | list[str], embed_size: int | None = None
    ) -> tuple[list[list[float]], UsageInfo]:
        ...
        texts: list[str]
        if isinstance(text, str):
            texts = [text]
        else:
            texts = text
        response = self.client.embeddings.create(
            model=self.model,
            input=texts,
            dimensions=embed_size if embed_size is not None else openai.omit,
        )
        usage_info = UsageInfo(response.usage.prompt_tokens, 0)
        output = [e.embedding for e in response.data]
        return output, usage_info
