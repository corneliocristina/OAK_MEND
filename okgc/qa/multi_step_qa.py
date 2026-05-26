###############################################################################
#
# Multi-step question answering method taken from:
#   Wikontic: Constructing Wikidata-Aligned, Ontology-Aware Knowledge Graphs with Large Language Models,
#   Alla Chepurova, Aydar Bulatov, Yuri Kuratov, Mikhail Burtsev, 2025
#
# okgc/qa.py
#
# Modified code originally from utils/structured_inference_with_db.py in
# https://github.com/screemix/Wikontic
#
###############################################################################

import os
from collections import defaultdict

from okgc.utils.filesystem import load_prompt
from okgc.utils.graph import Graph, Qualifier, Triple
from okgc.utils.openai_wrapper import OpenAIClient
from okgc.utils.sent_embed import SentenceEmbedder
from okgc.utils.sparql import Predicate, TypesHierarchy
from okgc.utils.typecheck import ontology_filter_triples, remove_qualifiers
from okgc.utils.vector_index import StrIndex


class MultiStepQA:
    def __init__(
        self,
        client: OpenAIClient,
        sent_embed: SentenceEmbedder,
        graph: Graph,
        *,
        use_qualifiers: bool = False,
        filter_triples: bool = False,
        filter_qualifiers: bool = False,
        types_hierarchy: TypesHierarchy,
        predicates: dict[str, Predicate],
        index_qualifiers_entities: bool = True,
        prompts_path: str | None = None,
        verbose: bool = False,
    ):
        if prompts_path is None:
            prompts_path = os.path.join("prompts", "wikontic", "qa")
        self.client = client
        self.sent_embed = sent_embed
        self.use_qualifiers = use_qualifiers
        self.filter_triples = filter_triples
        self.filter_qualifiers = filter_qualifiers
        self.index_qualifiers_entities = index_qualifiers_entities
        self.types_hierarchy = types_hierarchy
        self.predicates = predicates
        self.verbose = verbose

        # Filter triples (and possibly qualifiers) that violate the ontology
        if not self.use_qualifiers:
            graph = remove_qualifiers(graph)
        if self.filter_triples:
            graph = ontology_filter_triples(
                graph,
                hierarchy=self.types_hierarchy,
                filter_qualifiers=self.filter_qualifiers,
            )
        self.kg = graph

        # Build the entities index
        # Note that we also consider the entities aliases in the index, if any
        if index_qualifiers_entities:
            entities_index_data = (
                list(self.kg.entities)
                if graph.entity_aliases is None
                else graph.entity_aliases
            )
        else:
            entities: set[str] = set(s for s, _, _ in self.kg.triples) | set(
                o for _, _, o in self.kg.triples
            )
            if graph.entity_aliases is None:
                entities_index_data = list(entities)
            else:
                entities_index_data: dict[str, list[str]] = dict()
                for e in entities:
                    entities_index_data[e] = self.kg.entity_aliases[e]
        self.entities_index = StrIndex(entities_index_data, sent_embed=self.sent_embed)

        # Build the entity to relevant triple index
        self.relevant_triples_index = defaultdict(set)
        for t in self.kg.triples:
            s, p_id, o = t
            self.relevant_triples_index[s].add(t)
            self.relevant_triples_index[o].add(t)
            if self.kg.qualifiers is None or not index_qualifiers_entities:
                continue
            for q_id, q_o in self.kg.qualifiers.get(t, []):
                self.relevant_triples_index[q_o].add(t)

        self.prompts: dict[str, str] = {
            "question_decomposition": load_prompt(
                os.path.join(
                    prompts_path,
                    "question_decomposition_1.txt",
                )
            ),
            "qa_collapsing": load_prompt(
                os.path.join(prompts_path, "qa_collapsing_prompt.txt")
            ),
            "qa_is_answered": load_prompt(
                os.path.join(prompts_path, "prompt_is_answered.txt")
            ),
            "question_entity_extractor": load_prompt(
                os.path.join(prompts_path, "prompt_entity_extraction_from_question.txt")
            ),
            "qa": load_prompt(os.path.join(prompts_path, "qa_prompt.txt")),
        }

    def ask(
        self,
        question: str,
        *,
        max_num_triples: int = 150,
        max_entities_retrieval: int = 3,
        max_attempts: int = 5,
    ) -> str:
        # Decompose question
        collapsed_question = self.decompose_question(question)
        if self.verbose:
            print(f"Question: {question}")

        question_answer: str = ""
        question_sequence: list[str] = []
        answer_sequence: list[str] = []
        for i in range(max_attempts):
            # Extract entities from question
            extracted_entities = self.extract_entities_from_question(collapsed_question)
            if not extracted_entities:
                continue
            if question_answer:
                extracted_entities.append(question_answer)
            similar_entities = self.retrieve_similar_entity_names(
                extracted_entities,
                k=max_entities_retrieval,
                entities_index=self.entities_index,
            )

            # Retrieve all triples mentioning the extracted entities
            supporting_triples: list[Triple] = []
            for e in similar_entities:
                for t in self.relevant_triples_index[e]:
                    if t not in supporting_triples:
                        supporting_triples.append(t)
                if len(supporting_triples) >= max_num_triples:
                    supporting_triples = supporting_triples[:max_num_triples]
                    break

            supporting_qualifiers = None
            if self.kg.qualifiers is not None:
                supporting_qualifiers = {
                    t: self.kg.qualifiers[t]
                    for t in supporting_triples
                    if t in self.kg.qualifiers
                }

            question_answer = self.answer_question(
                collapsed_question,
                supporting_triples,
                supporting_qualifiers=supporting_qualifiers,
            )
            question_sequence.append(collapsed_question)
            answer_sequence.append(question_answer)

            if self.verbose:
                print(f"QA attempt #{i + 1}")
                print(f"One-hop question: {collapsed_question}")
                print(f"Relevant entities: {similar_entities}")
                print(f"Tentative answer: {question_answer}")
            answer_check = self.check_if_question_is_answered(
                question, question_sequence, answer_sequence
            )
            if "NOT FINAL" in answer_check:
                collapsed_question = self.collapse_question(
                    question, collapsed_question, question_answer
                )
            else:
                if self.verbose:
                    print(f"Final answer: {answer_check}")
                    print()
                return answer_check
        return question_answer

    def decompose_question(self, question: str) -> str:
        """Decompose a question using knowledge graph triplets."""
        decomposed_question, usage_info = self.client.get_completion(
            system_prompt=self.prompts["question_decomposition"],
            user_prompt=f"Question: {question}",
            transform_to_json=False,
        )
        assert isinstance(decomposed_question, str)
        return decomposed_question

    def collapse_question(
        self, original_question: str, question: str, answer: str
    ) -> str:
        """Collapse a question using knowledge graph triplets."""
        collapsed_question, usage_info = self.client.get_completion(
            system_prompt=self.prompts["qa_collapsing"],
            user_prompt=f"Original multi-hop question: {original_question}\nAnswered sub-question: {question}\nAnswer: {answer}",
            transform_to_json=False,
        )
        assert isinstance(collapsed_question, str)
        return collapsed_question

    def extract_entities_from_question(self, question: str) -> list[str]:
        """Extract entities from a question."""
        entities, usage_info = self.client.get_completion(
            system_prompt=self.prompts["question_entity_extractor"],
            user_prompt=f"Question: {question}",
            transform_to_json=True,
        )
        if isinstance(entities, (int, float, str)):
            entities = [entities]
        assert isinstance(entities, list), entities
        entities = [str(e) if isinstance(e, (int, float)) else e for e in entities]
        assert all(
            isinstance(e, str) or (isinstance(e, dict) and "name" in e)
            for e in entities
        ), entities
        entities = [e if isinstance(e, str) else e["name"] for e in entities]
        return entities

    def retrieve_similar_entity_names(
        self, entity_names: list[str], k: int = 3, *, entities_index: StrIndex
    ) -> list[str]:
        found_entities = entities_index.search(entity_names, k=k)
        assert isinstance(found_entities, list)
        result: list[str] = []
        for e, found_es in zip(entity_names, found_entities):
            if e in found_es:
                result.append(e)
                continue
            result.extend(found_es)
        return list(dict.fromkeys(result).keys())

    def answer_question(
        self,
        question: str,
        supporting_triples: list[Triple],
        *,
        supporting_qualifiers: dict[Triple, list[Qualifier]] | None = None,
    ) -> str:
        """Answer a question using knowledge graph triplets."""
        triplets: list[dict] = []
        if supporting_qualifiers is None:
            for t in supporting_triples:
                triplet = {
                    "subject": t[0],
                    "relation": self.kg.predicates[t[1]].name,
                    "object": t[2],
                }
                triplets.append(triplet)
        else:
            for t in supporting_triples:
                qs = []
                if t in supporting_qualifiers:
                    for q_id, q_o in supporting_qualifiers[t]:
                        q = {"relation": self.kg.predicates[q_id].name, "object": q_o}
                        qs.append(q)
                triplet = {
                    "subject": t[0],
                    "relation": self.kg.predicates[t[1]].name,
                    "object": t[2],
                    "qualifiers": qs,
                }
                triplets.append(triplet)
        triplets_repr = "\n".join(f"{tr}" for tr in triplets)
        answer, usage_info = self.client.get_completion(
            system_prompt=self.prompts["qa"],
            user_prompt=f"Question: {question}\n\nTriplets:\n{triplets_repr}",
            transform_to_json=False,
        )
        assert isinstance(answer, str)
        return answer

    def check_if_question_is_answered(
        self, question: str, subquestions: list[str], answers: list[str]
    ) -> str:
        """Check if a question is answered."""
        user_prompt = (
            f"Original multi-hop question: {question}\nQuestion -> answer sequence:\n"
        )
        for question, answer in zip(subquestions, answers):
            user_prompt += f"- {question} -> {answer}\n"
        answer, usage_info = self.client.get_completion(
            system_prompt=self.prompts["qa_is_answered"],
            user_prompt=user_prompt,
            transform_to_json=False,
        )
        assert isinstance(answer, str)
        return answer
