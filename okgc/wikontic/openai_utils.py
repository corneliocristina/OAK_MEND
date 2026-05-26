###############################################################################
#
# okgc/wikontic/openai_utils.py
#
# Code originally from src/wikontic/utils/openai_utils.py in
# https://github.com/screemix/Wikontic
# 70a4a79c8f54561012aebd4c4429736c22dad90b
#
# MIT License
#
###############################################################################

import json
import logging
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Union

import openai
import tenacity
from dotenv import find_dotenv, load_dotenv
from tenacity import (
    before_sleep_log,
    retry,
    stop_after_attempt,
    wait_random_exponential,
)

# Configure logging
logging.basicConfig(stream=sys.stderr)
logger = logging.getLogger("OpenAIUtils")
logger.setLevel(logging.ERROR)

_ = load_dotenv(find_dotenv())
# # OpenAI
# client = openai.OpenAI(
#     api_key=os.getenv("KEY"),
#     # http_client=http_client
# )

# OpenRouter
# base_url = os.getenv("OPENROUTER_BASE_URL")
# client = openai.OpenAI(base_url=base_url, api_key=os.getenv("OPENROUTER_KEY"))

# # Local inference
# client = openai.OpenAI(base_url=os.getenv("LOCAL_BASE_URL"),
#             api_key=os.getenv("LOCAL_KEY"),
#             http_client=openai.DefaultHttpxClient(verify=False)
#         )

MAX_ATTEMPTS = 1

reasoning_pattern = re.compile(r".*</think>", flags=re.DOTALL)


class LLMTripletExtractor:
    """A class for extracting and processing knowledge graph triplets using OpenAI's LLMs."""

    def __init__(
        self,
        prompt_folder_path: str = "./wikontic-prompts/",
        system_prompt_paths: Optional[Dict[str, str]] = None,
        model: str = "",
        base_url: str = "",
        api_key: str | None = None,
        max_attempts: int = 3,
        hotpot: bool = False,
        seed: int = 42,
    ):
        """
        Initialize the LLMTripletExtractor.

        Args:
            prompt_folder_path: Path to folder containing prompt files
            system_prompt_paths: Dictionary mapping prompt types to file paths
            model: Name of the OpenAI model to use
        """
        api_key = "" if api_key is None else api_key
        self.client = openai.OpenAI(base_url=base_url, api_key=api_key)
        if not model:
            model = self.client.models.list().data[0].id
        #
        self._avoid_system_prompt = any(m in model for m in ["DeepSeek-R1"])

        if system_prompt_paths is None:
            system_prompt_paths = {
                "triplet_extraction": "triplet_extraction/propmt_1_types_qualifiers.txt",
                # 'triplet_extraction': 'triplet_extraction/prompt_1_types_qualifiers_dialog_bench.txt',
                "relation_entity_types_ranker": "ontology_refinement/prompt_choose_relation_and_types.txt",
                "relation_ranker": "ontology_refinement/prompt_choose_relation.txt",
                "entity_types_ranker": "ontology_refinement/prompt_choose_entity_types.txt",
                "relation_ranker_wo_entity_types": "name_refinement/prompt_choose_relation_wo_entity_types.txt",
                # 'relation_ranker_wo_entity_types': 'name_refinement/prompt_choose_relation_wo_entity_types_dialog_bench.txt',
                # 'subject_ranker': 'name_refinement/rank_subject_names_dialog_bench.txt',
                "subject_ranker": "name_refinement/rank_subject_names.txt",
                # 'object_ranker': 'name_refinement/rank_object_names_dialog_bench.txt',
                "object_ranker": "name_refinement/rank_object_names.txt",
                "quailfier_object_ranker": "name_refinement/rank_object_qualifiers.txt",
                "question_entity_extractor": "qa/prompt_entity_extraction_from_question.txt",
                "question_entity_ranker": "qa/prompt_choose_relevant_entities_for_question.txt",
                "question_entity_ranker_wo_types": "qa/prompt_choose_relevant_entities_for_question_wo_types.txt",
                # 'qa': 'qa_prompt_hotpot.txt'
                "question_decomposition_1": "qa/question_decomposition_1.txt",
                "qa_collapsing": "qa/qa_collapsing_prompt.txt",
                "qa_is_answered": "qa/prompt_is_answered.txt",
                "qa": "qa/qa_prompt.txt",
            }
            if hotpot:
                system_prompt_paths["qa"] = "qa/qa_prompt_hotpot.txt"
            logger.info(f"System prompt qa paths: {system_prompt_paths['qa']}")

        # Load all prompts
        prompt_folder = Path(prompt_folder_path)
        self.prompts = {}
        for prompt_type, filename in system_prompt_paths.items():
            with open(prompt_folder / filename) as f:
                self.prompts[prompt_type] = f.read()

        self.model = model
        self.messages = []
        self.prompt_tokens_num = 0
        self.completion_tokens_num = 0
        self.seed = seed

        self._refine_attempt = 0
        self._prev_error = None  # store previous exception
        self.MAX_ATTEMPTS = max_attempts

    def extract_json(self, text: str) -> Union[dict, list, str]:
        """Extract JSON from text, handling both code blocks and inline JSON."""
        patterns = [
            r"```json\s*(\{.*?\}|\[.*?\])\s*```",  # JSON in code blocks
            r"(\{.*?\}|\[.*?\])",  # Inline JSON
        ]

        # print(text)

        try:
            # return json.loads(text)
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        for pattern in patterns:
            match = re.search(pattern, text, re.DOTALL)
            if match:
                try:
                    # return json.loads(match.group(1))
                    return json.loads(match.group(1))
                except json.JSONDecodeError:
                    logging.error(f"Failed to parse JSON: {text}")

        return text

    @retry(
        wait=wait_random_exponential(multiplier=1, max=60),
        before_sleep=before_sleep_log(logger, logging.ERROR),
        stop=stop_after_attempt(5),
    )
    def get_completion(
        self, system_prompt: str, user_prompt: str, transform_to_json: bool = True
    ) -> Union[dict, list, str]:
        """Get completion from OpenAI API with retry logic."""
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
        self.completion_tokens_num += response.usage.completion_tokens
        self.prompt_tokens_num += response.usage.prompt_tokens

        content = response.choices[0].message.content.strip()
        logger.debug("System prompt: %s\n%s", str(system_prompt), "-" * 100)
        logger.debug("User prompt: %s\n%s", str(user_prompt), "-" * 100)
        logger.debug("Output content: %s\n%s", str(content), "-" * 100)
        # print(content)

        # Remove thinking body, if any
        content = re.sub(reasoning_pattern, r"", content)

        output = self.extract_json(content) if transform_to_json else content

        self.messages = messages + [{"role": "assistant", "content": output}]
        return output

    @tenacity.retry(stop=tenacity.stop_after_attempt(MAX_ATTEMPTS), reraise=True)
    def extract_triplets_from_text(self, text: str) -> dict:
        """Extract knowledge graph triplets from text."""

        self._refine_attempt += 1
        attempt = self._refine_attempt
        logger.log(
            logging.DEBUG,
            "Attempt of a function call extract_triplets_from_text: %s",
            attempt,
        )
        system_prompt = self.prompts["triplet_extraction"]
        if attempt > 1:
            prev_error = self._prev_error
            system_prompt += f"\n(Previous attempt #{attempt - 1} failed with error: {prev_error}. Please adjust your answer!)"
            logger.log(logging.ERROR, "System prompt: %s", system_prompt)

        try:
            return self.get_completion(
                system_prompt=system_prompt, user_prompt=f'Text: "{text}"'
            )
        except Exception as e:
            self._prev_error = e
            # if json from output is broken after 3 attempts  - raise an exception
            logger.log(logging.ERROR, str(e))
            if attempt > self.MAX_ATTEMPTS:
                raise e

    @tenacity.retry(stop=tenacity.stop_after_attempt(MAX_ATTEMPTS), reraise=True)
    def refine_entity_types(
        self,
        text: str,
        triplet: dict,
        candidate_subject_types: List[str],
        candidate_object_types: List[str],
    ) -> dict:
        """Refine relations and entity types using candidate backbone triplets."""
        triplet_filtered = {
            k: triplet[k]
            for k in ["subject", "relation", "object", "subject_type", "object_type"]
        }

        candidates_subject_types_str = json.dumps(candidate_subject_types)
        candidates_object_types_str = json.dumps(candidate_object_types)
        logger.log(
            logging.DEBUG,
            "candidates subject types: %s\n%s",
            str(candidates_subject_types_str),
            "-" * 100,
        )
        logger.log(
            logging.DEBUG,
            "candidates object types: %s\n%s",
            str(candidates_object_types_str),
            "-" * 100,
        )

        self._refine_attempt += 1
        attempt = self._refine_attempt
        logger.log(
            logging.DEBUG, "Attempt of a function call refine_entity_types: %s", attempt
        )
        system_prompt = self.prompts["entity_types_ranker"]
        if attempt > 1:
            prev_error = self._prev_error
            system_prompt += f"\n(Previous attempt #{attempt - 1} failed with error: {prev_error}. Please adjust your answer!)"
            logger.log(logging.ERROR, "System prompt: %s", system_prompt)

        try:
            output = self.get_completion(
                system_prompt=system_prompt,
                user_prompt=f'Text: "{text}\nExtracted Triplet: {json.dumps(triplet_filtered)}\n'
                f"Candidate Subject Types: {candidates_subject_types_str}\n"
                f"Candidate Object Types: {candidates_object_types_str}",
            )
        except Exception as e:
            self._prev_error = e
            logger.log(logging.ERROR, str(e))
            # if json from output is broken after 3 attempts  - raise an exception
            if attempt > self.MAX_ATTEMPTS:
                raise e

        logger.log(
            logging.DEBUG,
            "refined subject type: %s\n%s",
            str(output["subject_type"]),
            "-" * 100,
        )
        logger.log(
            logging.DEBUG,
            "refined object type: %s\n%s",
            str(output["object_type"]),
            "-" * 100,
        )

        try:
            assert output["subject_type"] in candidate_subject_types, (
                "Refined subject type is not in candidate subject types"
            )
            assert output["object_type"] in candidate_object_types, (
                "Refined object type is not in candidate object types"
            )
        except Exception as e:
            self._prev_error = e
            logger.log(logging.ERROR, str(e))
            # do not raise an exception - save triplet in ontology filtered collection
        return output

    @tenacity.retry(stop=tenacity.stop_after_attempt(MAX_ATTEMPTS), reraise=True)
    def refine_relation(
        self, text: str, triplet: dict, candidate_relations: List[dict]
    ) -> dict:
        """Refine relation using candidate relations."""
        triplet_filtered = {
            k: triplet[k]
            for k in ["subject", "relation", "object", "subject_type", "object_type"]
        }

        candidates_str = json.dumps(candidate_relations, ensure_ascii=False)
        logger.log(
            logging.DEBUG,
            "candidates relations: %s\n%s",
            str(candidates_str),
            "-" * 100,
        )
        self._refine_attempt += 1
        attempt = self._refine_attempt

        logger.log(
            logging.DEBUG, "Attempt of a function call refine_relation: %s", attempt
        )
        system_prompt = self.prompts["relation_ranker"]

        if attempt > 1:
            prev_error = self._prev_error
            system_prompt += f"\n(Previous attempt #{attempt - 1} failed with error {prev_error}. Please adjust your answer!)"
            logger.log(logging.ERROR, "System prompt: %s", system_prompt)
        try:
            output = self.get_completion(
                system_prompt=system_prompt,
                user_prompt=f'Text: "{text}\nExtracted Triplet: {json.dumps(triplet_filtered, ensure_ascii=False)}\n'
                f"Candidate relations: {candidates_str}",
                transform_to_json=True,
            )
        except Exception as e:
            self._prev_error = e
            logger.log(logging.ERROR, str(e))
            # if json from output is broken after 3 attempts  - raise an exception
            if attempt > self.MAX_ATTEMPTS:
                raise e

        logger.log(
            logging.DEBUG,
            "refined relation: %s\n%s",
            str(output["relation"]),
            "-" * 100,
        )

        try:
            assert output["relation"] in candidate_relations, (
                "Refined relation is not in candidate relations"
            )
        except Exception as e:
            self._prev_error = e
            logger.log(logging.ERROR, str(e))
            # do not raise an exception - save triplet in ontology filtered collection

        return output

    @tenacity.retry(stop=tenacity.stop_after_attempt(MAX_ATTEMPTS), reraise=True)
    def refine_relation_wo_entity_types(
        self, text: str, triplet: dict, candidate_relations: List[dict]
    ) -> dict:
        """Refine relation using candidate relations."""
        triplet_filtered = {k: triplet[k] for k in ["subject", "relation", "object"]}
        candidates_str = json.dumps(candidate_relations, ensure_ascii=False)
        logger.log(
            logging.DEBUG,
            "candidates relations: %s\n%s",
            str(candidates_str),
            "-" * 100,
        )

        attempt = self._refine_attempt

        logger.log(
            logging.DEBUG,
            "Attempt of a function call refine_relation_wo_entity_types: %s",
            attempt,
        )
        self._refine_attempt += 1
        system_prompt = self.prompts["relation_ranker_wo_entity_types"]

        if attempt > 1:
            prev_error = self._prev_error
            system_prompt += f"\n(Previous attempt #{attempt - 1} failed with error {prev_error}. Please adjust your answer!)"
            logger.log(logging.ERROR, "System prompt: %s", system_prompt)
        try:
            return self.get_completion(
                system_prompt=system_prompt,
                user_prompt=f'Text: "{text}\nExtracted Triplet: {json.dumps(triplet_filtered, ensure_ascii=False)}\n'
                f"Candidate relations: {candidates_str}",
                transform_to_json=False,
            )
        except Exception as e:
            self._prev_error = e
            logger.log(logging.ERROR, str(e))
            # if json from output is broken after 3 attempts  - raise an exception
            if self._refine_attempt > self.MAX_ATTEMPTS:
                raise e

    def refine_relation_and_entity_types(
        self, text: str, triplet: dict, candidate_triplets: List[dict]
    ) -> dict:
        """Refine relations and entity types using candidate backbone triplets."""
        triplet_filtered = {
            k: triplet[k]
            for k in ["subject", "relation", "object", "subject_type", "object_type"]
        }

        candidates_str = "".join(f"{json.dumps(c)}\n" for c in candidate_triplets)

        return self.get_completion(
            system_prompt=self.prompts["relation_entity_types_ranker"],
            user_prompt=f'Text: "{text}\nExtracted Triplet: {json.dumps(triplet_filtered)}\n'
            f"Candidate Triplets: {candidates_str}",
        )

    def refine_entity(
        self,
        text: str,
        triplet: dict,
        candidates: List[str],
        is_object: bool = False,
        role: str = "user",
    ) -> dict:
        """Refine subject/object names using candidate options from pre-built KG."""

        triplet_filtered = {k: triplet[k] for k in ["subject", "relation", "object"]}
        original_name = triplet_filtered["object" if is_object else "subject"]

        self._refine_attempt += 1
        attempt = self._refine_attempt

        logger.log(
            logging.DEBUG, "Attempt of a function call refine_entity: %s", attempt
        )
        prompt_key = "object_ranker" if is_object else "subject_ranker"
        entity_type = "Object" if is_object else "Subject"
        system_prompt = self.prompts[prompt_key]

        if attempt > 1:
            prev_error = self._prev_error
            system_prompt += f"\n(Previous attempt #{attempt - 1} failed with error: {prev_error}. Please adjust your answer!)"
            logger.log(logging.ERROR, "System prompt: %s", system_prompt)

        try:
            return self.get_completion(
                system_prompt=system_prompt,
                user_prompt=f'Text: "{text}\nRole: {role}\nExtracted Triplet: {json.dumps(triplet_filtered, ensure_ascii=False)}\n'
                f"Original {entity_type}: {original_name}\n"
                f'Candidate {entity_type}s: {json.dumps(candidates, ensure_ascii=False)}"',
                transform_to_json=False,
            )
        except Exception as e:
            self._prev_error = e
            logger.log(logging.ERROR, str(e))
            # if json from output is broken after 3 attempts  - raise an exception
            if attempt > self.MAX_ATTEMPTS:
                raise e

    def extract_entities_from_question(self, question: str) -> dict:
        """Extract entities from a question."""
        return self.get_completion(
            system_prompt=self.prompts["question_entity_extractor"],
            user_prompt=f"Question: {question}",
        )

    def identify_relevant_entities(
        self, question: str, entity_list: List[str]
    ) -> List[str]:
        """Identify entities relevant to a question."""
        return self.get_completion(
            system_prompt=self.prompts["question_entity_ranker"],
            user_prompt=f"Question: {question}\nEntities: {entity_list}",
        )

    def identify_relevant_entities_wo_types(
        self, question: str, entity_list: List[str]
    ) -> List[str]:
        """Identify entities relevant to a question."""
        return self.get_completion(
            system_prompt=self.prompts["question_entity_ranker_wo_types"],
            user_prompt=f"Question: {question}\nEntities: {entity_list}",
        )

    def answer_question(self, question: str, triplets: List[dict]) -> str:
        """Answer a question using knowledge graph triplets."""
        return self.get_completion(
            system_prompt=self.prompts["qa"],
            user_prompt=f'Question: {question}\n\nTriplets: "{triplets}"',
            transform_to_json=False,
        )

    def collapse_question(
        self, original_question: str, question: str, answer: str
    ) -> str:
        """Collapse a question using knowledge graph triplets."""
        return self.get_completion(
            system_prompt=self.prompts["qa_collapsing"],
            user_prompt=f"Original multi-hop question: {original_question}\n\Answered sub-question: {question}\n\Answer: {answer}",
            transform_to_json=True,
        )

    def decompose_question(self, question: str) -> str:
        """Decompose a question using knowledge graph triplets."""
        return self.get_completion(
            system_prompt=self.prompts["question_decomposition_1"],
            user_prompt=f"Question: {question}",
            transform_to_json=False,
        )

    def check_if_question_is_answered(
        self, question: str, subquestions: List[str], answers: List[str]
    ) -> str:
        """Check if a question is answered."""
        user_prompt = (
            f"Original multi-hop question: {question}\nQuestion->answer sequence:\n"
        )
        for question, answer in zip(subquestions, answers):
            user_prompt += f"{question} -> {answer}\n"
        return self.get_completion(
            system_prompt=self.prompts["qa_is_answered"],
            user_prompt=user_prompt,
            transform_to_json=False,
        )

    def calculate_used_tokens(self) -> int:
        """Calculate the total # of used tokens for generation"""
        return self.prompt_tokens_num, self.completion_tokens_num

    def reset_tokens(self):
        """Reset the total # of used tokens for generation"""
        self.prompt_tokens_num = 0
        self.completion_tokens_num = 0

    def reset_messages(self):
        """Reset the messages"""
        self.messages = []

    def reset_error_state(self):
        self._prev_error = None
        self._refine_attempt = 0
