import functools
import os
from collections import defaultdict
from typing import Any

import numpy as np
import spacy
from rank_bm25 import BM25Okapi
from sklearn.cluster import KMeans

from okgc.utils.aliases import spacy_find_entity_aliases, spacy_nlp_tokenize
from okgc.utils.filesystem import load_prompt
from okgc.utils.graph import Graph, Qualifier, Triple, deduplicate_entities
from okgc.utils.openai_wrapper import OpenAIClient
from okgc.utils.sent_embed import SentenceEmbedder
from okgc.utils.sparql import Predicate, TypeInfo
from okgc.utils.strings import normalize_lowercase
from okgc.utils.typecheck import (
    satisfy_domain_range_constraint,
    satisfy_qualifier_constraints,
)
from okgc.utils.usage import UsageInfo
from okgc.utils.vector_index import PredicatesVectorIndex, StrIndex, TypesVectorIndex


class OAK:
    def __init__(
        self,
        client: OpenAIClient,
        sent_embed: SentenceEmbedder,
        *,
        types_index: TypesVectorIndex,
        predicates_index: PredicatesVectorIndex,
        qualifier_predicates_index: PredicatesVectorIndex,
        canonicalize_type_method: str = "llm",
        canonicalize_predicate_method: str = "llm",
        canonicalize_threshold: float = 0.05,
        consistency_rerank: bool = False,
        consistency_rerank_beta: float = 0.5,
        prompts_path: str | None = None,
        verbose: bool = False,
    ):
        assert canonicalize_type_method in ["embedding", "llm"]
        assert canonicalize_predicate_method in ["embedding", "llm"]
        if prompts_path is None:
            prompts_path = os.path.join("prompts", "oak-mend")
        self.client = client
        self.sent_embed = sent_embed
        self.types_index = types_index
        self.predicates_index = predicates_index
        self.qualifier_predicates_index = qualifier_predicates_index
        self.canonicalize_type_method = canonicalize_type_method
        self.canonicalize_predicate_method = canonicalize_predicate_method
        self.canonicalize_threshold = canonicalize_threshold
        self.consistency_rerank = consistency_rerank
        self.consistency_rerank_beta = consistency_rerank_beta
        self.verbose = verbose
        self._nlp: spacy.Language | None = None

        self.prompts: dict[str, str] = {
            "extract_triples": load_prompt(
                os.path.join(prompts_path, "extract_triples_qualifiers_1shot.txt")
            ),
            "extract_triples_fallback": load_prompt(
                os.path.join(
                    prompts_path, "extract_triples_qualifiers_1shot_fallback.txt"
                )
            ),
            "canonicalize_type": load_prompt(
                os.path.join(prompts_path, "canonicalize_type.txt")
            ),
            "canonicalize_predicate": load_prompt(
                os.path.join(prompts_path, "canonicalize_predicate.txt")
            ),
            "deduplicate_entities": load_prompt(
                os.path.join(prompts_path, "deduplicate_entities.txt")
            ),
        }

    def generate_from_text(
        self, text: str, *, text_id: int = 0
    ) -> tuple[Graph, dict[str, UsageInfo]]:
        # (0) Extract raw triples and qualifiers from text
        (
            raw_triples,
            raw_predicates,
            raw_qualifier_predicates,
            raw_entity_types,
            raw_qualifiers,
            ext_triples_usage,
        ) = self.extract_raw_triples_and_qualifiers_from_text(text)

        # (1) Canonicalize entity types and predicates
        entity_types, canon_types_usage = self.canonicalize_entity_types(
            raw_entity_types,
            text=text,
            threshold=self.canonicalize_threshold,
        )
        if self.verbose:
            print("Canonicalized types:")
            for e, ts in entity_types.items():
                print(e, "=>", [t.name for t in ts])
            print()
        predicates, canon_predicates_usage = self.canonicalize_predicates(
            raw_predicates,
            raw_triples,
            entity_types=entity_types,
            text=text,
            threshold=self.canonicalize_threshold,
        )
        if self.verbose:
            print("Canonicalized predicates:")
            for r, p in predicates.items():
                print(r, "=>", p.name)
            print()
        qualifier_predicates = self.canonicalize_qualifier_predicates(
            raw_qualifier_predicates,
            raw_qualifiers,
            predicates=predicates,
            entity_types=entity_types,
        )
        if self.verbose:
            print("Canonicalized qualifier predicates:")
            for r, p in qualifier_predicates.items():
                print(r, "=>", p.name)
            print()

        # (2) Construct the knowledge graph
        entities = set(entity_types.keys())
        entity_text_ids = {e: [text_id] for e in entity_types.keys()}
        out = Graph(
            entities=entities,
            entity_types=entity_types,
            entity_descriptions=None,
            predicates={},
            triples=set(),
            entity_text_ids=entity_text_ids,
            triple_text_ids={},
            qualifiers={},
        )
        for raw_triple in raw_triples:
            s, r, o = raw_triple
            p = predicates[r]

            # If the canonicalized predicate does not satisfy domain-range constraints,
            # then keep the original predicate as is (i.e., as an unknown predicate)
            # By doing so, we keep track of the original predicate name useful later for the corrections phase
            ok_subject, ok_object = satisfy_domain_range_constraint(
                p,
                entity_types[s],
                entity_types[o],
                hierarchy=self.types_index.hierarchy,
            )
            if not ok_subject or not ok_object:
                p = Predicate.unknown_from_name(r, candidate_code=p.code)
            # Add the triple and the predicate to the knowledge graph
            triple = (s, p.code, o)
            out.triples.add(triple)
            out.triple_text_ids[triple] = [text_id]
            out.predicates[p.code] = p

            # Collect the triple qualifiers and check whether they are consistent
            for q_r, q_o in raw_qualifiers[raw_triple]:
                q_p = qualifier_predicates[q_r]
                assert out.qualifiers is not None
                if triple not in out.qualifiers:
                    out.qualifiers[triple] = []
                # Similarly to the above for triples, if the canonicalize qualifier predicate does not
                # satisfy the qualifier constraints, then keep the original qualifier predicate as is
                ok_predicate, ok_object = satisfy_qualifier_constraints(
                    p, q_p, entity_types[q_o], hierarchy=self.types_index.hierarchy
                )
                if not ok_predicate or not ok_object:
                    q_p = Predicate.unknown_from_name(q_r, candidate_code=q_p.code)
                # Add the qualifier to the knowledge graph
                q = (q_p.code, q_o)
                out.qualifiers[triple].append(q)
                out.predicates[q_p.code] = q_p

        if self.verbose:
            print("Generated triples:")
            for triple in out.triples:
                s, p_id, o = triple
                labeled_triple = (s, out.predicates[p_id].name, o)
                print("Triple:", labeled_triple)
                print("Subject:", s, [t.name for t in out.entity_types[s]])
                print("Object:", o, [t.name for t in out.entity_types[o]])
                assert out.qualifiers is not None
                labeled_qualifiers = [
                    (out.predicates[q_id].name, q_o)
                    for q_id, q_o in out.qualifiers.get(triple, [])
                ]
                print("Qualifiers:", labeled_qualifiers)
                print()
            print()

        usage_info = {
            "extract_triples": ext_triples_usage,
            "canonicalize_types": canon_types_usage,
            "canonicalize_predicates": canon_predicates_usage,
        }
        return out, usage_info

    def deduplicate_entities(
        self, kg: Graph, *, method: str = "llm"
    ) -> tuple[Graph, UsageInfo]:
        entity_aliases: dict[str, list[str]]
        if method == "spacy":
            entity_aliases = self.deduplicate_entities_spacy(kg)
            usage_info = UsageInfo()
        elif method == "llm":
            entity_aliases, usage_info = self.deduplicate_entities_llm(kg)
        else:
            raise ValueError(f"Unknown canonicalization method based on '{method}'")
        return deduplicate_entities(kg, aliases=entity_aliases), usage_info

    def parse_raw_triples(
        self, raw_output: list[dict[str, Any]], *, fallback: bool = False
    ) -> tuple[
        set[Triple],
        set[str],
        set[str],
        dict[str, list[str]],
        dict[Triple, list[tuple[str, str]]],
    ]:
        triples: set[Triple] = set()
        predicates: set[str] = set()
        qualifier_predicates: set[str] = set()
        entity_types: dict[str, list[str]] = defaultdict(list)
        qualifiers: dict[Triple, list[tuple[str, str]]] = defaultdict(list)
        for entry in raw_output:
            # Normalize the JSON output
            if fallback:
                assert (
                    "subject" in entry and "predicate" in entry and "object" in entry
                ), entry
                s = entry["subject"]
                r = entry["predicate"]
                o = entry["object"]
            else:
                assert len(entry["triple"]) == 3, entry
                s, r, o = tuple(entry["triple"])
            r = normalize_lowercase(r)
            s, o = str(s), str(o)
            triple = (s, r, o)
            subject_type = entry["subject_type"]
            object_type = entry["object_type"]
            subject_type = normalize_lowercase(subject_type)
            object_type = normalize_lowercase(object_type)
            if fallback:
                qs = [
                    (q["predicate"], q["object"], q["object_type"])
                    for q in entry["qualifiers"]
                ]
            else:
                qs = [
                    (*q["pair"], q["object_type"]) for q in entry.get("qualifiers", [])
                ]

            # Collect triples, the entity types and the triple qualifiers
            predicates.add(r)
            if subject_type not in entity_types[s]:
                entity_types[s].append(subject_type)
            if object_type not in entity_types[o]:
                entity_types[o].append(object_type)
            for q_r, q_o, q_object_type in qs:
                q_r = normalize_lowercase(q_r)
                qualifier_predicates.add(q_r)
                q_object_type = normalize_lowercase(q_object_type)
                if q_object_type not in entity_types[q_o]:
                    entity_types[q_o].append(q_object_type)
                q = (q_r, q_o)
                if q not in qualifiers[triple]:
                    qualifiers[triple].append(q)
            triples.add(triple)

        return (
            triples,
            predicates,
            qualifier_predicates,
            entity_types,
            qualifiers,
        )

    def extract_raw_triples_and_qualifiers_from_text(
        self, text: str
    ) -> tuple[
        set[Triple],
        set[str],
        set[str],
        dict[str, list[str]],
        dict[Triple, list[Qualifier]],
        UsageInfo,
    ]:
        # Get completion
        user_prompt = f"""### Input\nText:\n{text}\n"""
        raw_output, usage_info = self.client.get_completion(
            self.prompts["extract_triples"], user_prompt, transform_to_json=True
        )
        assert isinstance(raw_output, list)

        try:
            parse_output = self.parse_raw_triples(raw_output)
        except:
            # Get completion (fallback prompt)
            raw_output, usage_info = self.client.get_completion(
                self.prompts["extract_triples_fallback"],
                user_prompt,
                transform_to_json=True,
            )
            assert isinstance(raw_output, list)
            parse_output = self.parse_raw_triples(raw_output, fallback=True)

        triples, predicates, qualifier_predicates, entity_types, qualifiers = (
            parse_output
        )
        return (
            triples,
            predicates,
            qualifier_predicates,
            entity_types,
            qualifiers,
            usage_info,
        )

    def canonicalize_entity_types(
        self,
        raw_entity_types: dict[str, list[str]],
        *,
        text: str,
        max_num_entities: int = 5,
        max_num_candidates: int = 25,
        threshold: float = 0.15,
    ) -> tuple[dict[str, list[TypeInfo]], UsageInfo]:
        if not raw_entity_types:
            return {}, UsageInfo()
        # Search for types in the index
        raw_types: list[str] = list(
            set(raw_t for raw_ts in raw_entity_types.values() for raw_t in raw_ts)
        )
        if not raw_types:
            return {}, UsageInfo()
        search_output = self.types_index.search(
            raw_types, k=max_num_candidates, return_scores=True
        )
        assert isinstance(search_output, tuple)
        found_types, types_scores = search_output

        # Assign the types
        usage_info = UsageInfo()
        types: dict[str, TypeInfo] = {}
        for raw_t, ts, ss in zip(raw_types, found_types, types_scores):
            ents = [e for e, raw_ts in raw_entity_types.items() if raw_t in raw_ts]
            t, usage = self.retrieve_type_from_candidates(
                raw_t,
                ents,
                ts,
                ss,
                text=text,
                max_num_entities=max_num_entities,
                threshold=threshold,
            )
            types[raw_t] = t
            usage_info += usage

        # Assign the extracted types to entities
        entity_types = {
            e: [types[raw_t] for raw_t in raw_ts]
            for e, raw_ts in raw_entity_types.items()
        }
        return entity_types, usage_info

    def canonicalize_predicates(
        self,
        raw_predicates: set[str],
        raw_triples: set[Triple],
        *,
        entity_types: dict[str, list[TypeInfo]],
        text: str,
        num_retrieval_predicates: int = 100,
        max_num_candidates: int = 25,
        threshold: float = 0.15,
    ) -> tuple[dict[str, Predicate], UsageInfo]:
        # Search for predicates in the index
        raw_predicates = list(raw_predicates)
        if not raw_predicates:
            return {}, UsageInfo()
        search_output = self.predicates_index.search(
            raw_predicates, k=num_retrieval_predicates, return_scores=True
        )
        assert isinstance(search_output, tuple)
        found_predicates, predicates_scores = search_output

        # Assign the predicates
        usage_info = UsageInfo()
        predicates: dict[str, Predicate] = {}
        for raw_p, ps, ss in zip(raw_predicates, found_predicates, predicates_scores):
            entity_pairs = [(s, o) for s, r, o in raw_triples if r == raw_p]
            predicates[raw_p], usage = self.retrieve_predicate_from_candidates(
                raw_p,
                entity_pairs,
                ps,
                ss,
                entity_types=entity_types,
                text=text,
                max_num_candidates=max_num_candidates,
                threshold=threshold,
            )
            usage_info += usage
        return predicates, usage_info

    def canonicalize_qualifier_predicates(
        self,
        raw_predicates: set[str],
        raw_qualifiers: dict[Triple, list[Qualifier]],
        *,
        predicates: dict[str, Predicate],
        entity_types: dict[str, list[TypeInfo]],
        num_retrieval_predicates: int = 25,
    ) -> dict[str, Predicate]:
        # Search for predicates in the index
        raw_predicates = list(raw_predicates)
        if not raw_predicates:
            return {}
        search_output = self.qualifier_predicates_index.search(
            raw_predicates, k=num_retrieval_predicates, return_scores=True
        )
        assert isinstance(search_output, tuple)
        found_predicates, predicates_scores = search_output

        # Assign the predicates
        q_predicates: dict[str, Predicate] = {}
        for raw_p, ps, ss in zip(raw_predicates, found_predicates, predicates_scores):
            predicate_object_pairs = [
                (r, q_o)
                for (s, r, o), qs in raw_qualifiers.items()
                for (q_r, q_o) in qs
                if q_r == raw_p
            ]
            q_predicates[raw_p] = self.retrieve_qualifier_predicate_from_candidates(
                raw_p,
                predicate_object_pairs,
                ps,
                ss,
                predicates=predicates,
                entity_types=entity_types,
            )
        return q_predicates

    def retrieve_type_from_candidates(
        self,
        name: str,
        entities: list[str],
        candidates: list[TypeInfo],
        candidate_scores: list[float],
        *,
        text: str,
        max_num_entities: int = 5,
        threshold: float = 0.15,
    ) -> tuple[TypeInfo, UsageInfo]:
        # If there is a predicate with the same exact name, then use that
        t_exact = next((t for t in candidates if t.name == name), None)
        if t_exact is not None:
            return t_exact, UsageInfo()
        # Otherwise, pick the highest scoring one
        max_score = candidate_scores[0]
        candidates = [
            t for t, s in zip(candidates, candidate_scores) if s > max_score - threshold
        ]
        assert candidates
        if len(candidates) == 1 or self.canonicalize_type_method == "embedding":
            return candidates[0], UsageInfo()
        assert self.canonicalize_type_method == "llm"
        entities = entities[:max_num_entities]
        entities_repr = "\n".join(f"- {e}" for e in entities)
        candidate_types_names = [t.name for t in candidates]
        candidate_types_names_fallback = [t.basename() for t in candidates]
        candidate_types_repr = "\n".join(f"- {t.name}" for t in candidates)
        candidate_types_map = dict(zip(candidate_types_names, candidates))
        candidate_types_map_fallback = dict(
            zip(candidate_types_names_fallback, candidates)
        )
        if self.verbose:
            print(
                f"Asking the LLM to disambiguate the type '{name}' for entities {entities}"
            )
            print(f"Candidates: {candidate_types_names}")
        # If multiple types have the same score (e.g., because they share an alias),
        # then ask the LLM to select the most suitable entity type
        user_prompt = f"""### Input\nText:\n{text}\n\n"""
        user_prompt += f"""Entities:\n{entities_repr}\n\n"""
        user_prompt += f"""Type: {name}\n\n"""
        user_prompt += f"""Candidates:\n{candidate_types_repr}\n"""
        selected_type, usage_info = self.client.get_completion(
            self.prompts["canonicalize_type"], user_prompt, transform_to_json=False
        )
        assert isinstance(selected_type, str)
        if self.verbose:
            print(f"LLM >> '{selected_type}'")
            print()
        if selected_type in candidate_types_map:
            return candidate_types_map[selected_type], usage_info
        if selected_type in candidate_types_map_fallback:
            return candidate_types_map_fallback[selected_type], usage_info
        return candidates[0], usage_info

    def retrieve_predicate_from_candidates(
        self,
        name: str,
        entity_pairs: list[tuple[str, str]],
        candidates: list[Predicate],
        candidate_scores: list[float],
        *,
        entity_types: dict[str, list[TypeInfo]],
        text: str,
        max_num_candidates: int = 25,
        threshold: float = 0.15,
    ) -> tuple[Predicate, UsageInfo]:
        def _compute_ontology_consistency_score(
            p: Predicate, *, embed_score: float
        ) -> float:
            num_consistent = 0
            for s, o in entity_pairs:
                ok_s, ok_o = satisfy_domain_range_constraint(
                    p,
                    entity_types[s],
                    entity_types[o],
                    hierarchy=self.types_index.hierarchy,
                )
                if ok_s and ok_o:
                    num_consistent += 1
                    continue
                ok_s_rev, ok_o_rev = satisfy_domain_range_constraint(
                    p,
                    entity_types[o],
                    entity_types[s],
                    hierarchy=self.types_index.hierarchy,
                )
                if ok_s_rev and ok_o_rev:
                    num_consistent += 1
            return (
                1.0 - self.consistency_rerank_beta
            ) * embed_score + self.consistency_rerank_beta * num_consistent / len(
                entity_pairs
            )

        # If there is a predicate with the same exact name, then use that
        p_exact = next((p for p in candidates if p.name == name), None)
        if p_exact is not None:
            return p_exact, UsageInfo()
        # Apply re-ranking using ontology consistency score
        if self.consistency_rerank:
            reranked_candidates = sorted(
                [
                    (p, _compute_ontology_consistency_score(p, embed_score=s))
                    for p, s in zip(candidates, candidate_scores)
                ],
                key=lambda x: x[1],
                reverse=True,
            )
            reranked_candidates = reranked_candidates[:max_num_candidates]
            candidates, candidate_scores = tuple(zip(*reranked_candidates))
        else:
            candidates = candidates[:max_num_candidates]
            candidate_scores = candidate_scores[:max_num_candidates]

        # If multiple predicates have the same score (e.g., because they share an alias),
        # then pick the predicate having name with the lowest embedding distance
        max_score = candidate_scores[0]
        candidates = [
            p for p, s in zip(candidates, candidate_scores) if s > max_score - threshold
        ]
        assert candidates
        if len(candidates) == 1 or self.canonicalize_predicate_method == "embedding":
            return candidates[0], UsageInfo()

        assert self.canonicalize_type_method == "llm"
        entity_pairs_repr = "\n".join(f"- {ep}" for ep in entity_pairs)
        candidate_predicates_names = [p.name for p in candidates]
        candidate_predicates_repr = "\n".join(f"- {p.name}" for p in candidates)
        candidate_predicates_map = dict(zip(candidate_predicates_names, candidates))
        if self.verbose:
            print(
                f"Asking the LLM to disambiguate the predicate '{name}' for entity pairs {entity_pairs}"
            )
            print(f"Candidates: {candidate_predicates_names}")
        # If multiple predicates have the same score (e.g., because they share an alias),
        # then ask the LLM to select the most suitable predicate
        user_prompt = f"""### Input\nText:\n{text}\n\n"""
        user_prompt += f"""Entity pairs:\n{entity_pairs_repr}\n\n"""
        user_prompt += f"""Predicate: {name}\n\n"""
        user_prompt += f"""Candidates:\n{candidate_predicates_repr}\n"""
        selected_predicate, usage_info = self.client.get_completion(
            self.prompts["canonicalize_predicate"], user_prompt, transform_to_json=False
        )
        assert isinstance(selected_predicate, str)
        if self.verbose:
            print(f"LLM >> '{selected_predicate}'")
            print()
        if selected_predicate in candidate_predicates_map:
            return candidate_predicates_map[selected_predicate], usage_info
        return candidates[0], usage_info

    def retrieve_qualifier_predicate_from_candidates(
        self,
        name: str,
        predicate_object_pairs: list[tuple[str, str]],
        candidates: list[Predicate],
        candidate_scores: list[float],
        *,
        predicates: dict[str, Predicate],
        entity_types: dict[str, list[TypeInfo]],
    ):
        def _compute_ontology_consistency_score(
            p: Predicate, *, embed_score: float
        ) -> float:
            num_consistent = 0
            for r, q_o in predicate_object_pairs:
                ok_r, ok_q_o = satisfy_qualifier_constraints(
                    predicates[r],
                    p,
                    entity_types[q_o],
                    hierarchy=self.types_index.hierarchy,
                )
                if ok_r and ok_q_o:
                    num_consistent += 1
            return (
                1.0 - self.consistency_rerank_beta
            ) * embed_score + self.consistency_rerank_beta * num_consistent / len(
                predicate_object_pairs
            )

        # If there is a predicate with the same exact name, then use that
        p_exact = next((p for p in candidates if p.name == name), None)
        if p_exact is not None:
            return p_exact
        # Apply re-ranking using ontology consistency score
        if self.consistency_rerank:
            reranked_candidates = sorted(
                [
                    (p, _compute_ontology_consistency_score(p, embed_score=s))
                    for p, s in zip(candidates, candidate_scores)
                ],
                key=lambda x: x[1],
                reverse=True,
            )
            candidates, candidate_scores = tuple(zip(*reranked_candidates))
        # Return the highest scoring qualifier predicate
        return candidates[0]

    def deduplicate_entities_spacy(self, kg: Graph) -> dict[str, list[str]]:
        if self._nlp is None:
            self._nlp = spacy.load("en_core_web_sm")
        return spacy_find_entity_aliases(kg, nlp=self._nlp)

    def deduplicate_entities_llm(
        self,
        kg: Graph,
        *,
        embed_size: int | None = None,
        avg_cluster_size: int = 128,
        max_num_candidates: int = 16,
        threshold: float = 0.9,
        use_bm25: bool = False,
    ) -> tuple[dict[str, list[str]], UsageInfo]:
        # Adapted from KGGen's entities deduplication algorithm using BM25 + embeddings + LLM calls
        # https://github.com/stair-lab/kg-gen/blob/main/src/kg_gen/utils/llm_deduplicate.py
        # https://arxiv.org/abs/2502.09956
        #
        # Main changes:
        # - Using our data structures and embedding models, and without using DSPy for constructing the prompts
        # - We do not care about deduplication of predicates, as we canonicalize to Wikidata instead
        # - We skip the deduplication step for entities that are literals, such as a numerical quantity or a calendar date
        #   We do so by exploiting the types associated to the entities. By doing so, we reduce the number of LLM calls for entity deduplication
        #
        assert avg_cluster_size > max_num_candidates
        if use_bm25:
            if self._nlp is None:
                self._nlp = spacy.load("en_core_web_sm")
            tokenize_func = functools.partial(spacy_nlp_tokenize, nlp=self._nlp)
            threshold = float("-inf")

        # Embed the entities
        entities = list(kg.entities)
        num_entities = len(entities)
        embeddings = self.sent_embed.encode(entities, embed_size=embed_size)
        embeddings = embeddings.numpy().astype(np.float32, copy=False)

        # Cluster the entities embeddings with KMeans
        num_clusters = max(1, num_entities // avg_cluster_size)
        kmeans = KMeans(
            n_clusters=num_clusters,
            init="random",
            n_init=1,
            max_iter=50,
            algorithm="lloyd",
            random_state=42,
        )
        kmeans.fit(embeddings)
        ys = kmeans.predict(embeddings)
        clusters: list[list[int]] = []
        for i in range(num_clusters):
            clusters.append(np.argwhere(ys == i).flatten().tolist())

        # Use the LLM to deduplicate entities independently within each cluster
        entity_aliases: dict[str, list[str]] = {}
        usage_info = UsageInfo()
        for i, cs in enumerate(clusters, start=1):
            # If the type of the entity is a literal, then skip the deduplication step
            es = []
            for c in cs:
                e = entities[c]
                if any(
                    self.types_index.hierarchy.is_literal(t) for t in kg.entity_types[e]
                ):
                    entity_aliases[e] = [e]
                    continue
                es.append(e)
            if not es:
                continue
            # Build an embedding index for the remaining entities
            index = StrIndex(es, sent_embed=self.sent_embed)
            if use_bm25:
                # Build the BM25 ranker object as well
                bm25_ranker = BM25Okapi(es, tokenizer=tokenize_func)
            if self.verbose:
                print(f"Processing entities cluster #{i}: {es}")
            remaining = set(es)
            while remaining:
                target = remaining.pop()

                # Set default values fo alias and duplicates, i.e., no duplicates have been found
                alias = target
                duplicates = [target]

                # Search for candidates using sentence embeddings and BM25 scores
                search_output = index.search(target, k=len(es), return_scores=True)
                assert isinstance(search_output, tuple)
                candidates, embedding_scores = search_output
                candidates, embedding_scores = candidates[0], embedding_scores[0]
                es_scores: dict[str, float] = {
                    e: s for e, s in zip(candidates, embedding_scores)
                }
                if use_bm25:
                    bm25_scores = bm25_ranker.get_scores(tokenize_func(target))
                    # Compute the combined scores
                    for i, e in enumerate(es):
                        es_scores[e] = 0.5 * es_scores[e] + 0.5 * bm25_scores[i]
                # Retrieve the entities that are candidate for deduplication,
                # i.e., whose score is greater than a threshold, and they have not already been deduplicated
                sorted_es_scores = sorted(
                    es_scores.items(), key=lambda x: x[1], reverse=True
                )
                candidates = [
                    e for e, s in sorted_es_scores if s > threshold and e in remaining
                ]
                # Keep only a maximum number of candidates
                candidates = candidates[:max_num_candidates]

                # Prompt the LLM to find duplicates in the cluster
                if candidates:
                    user_prompt = f"""### Input\nEntity: {target}\n\n"""
                    user_prompt += f"""Candidates: {candidates}\n"""
                    response, usage = self.client.get_completion(
                        self.prompts["deduplicate_entities"],
                        user_prompt,
                        transform_to_json=True,
                    )
                    assert isinstance(response, dict)
                    usage_info += usage
                    if response and response["alias"] and response["duplicates"]:
                        duplicates = response["duplicates"]
                        alias = response["alias"]
                        assert isinstance(duplicates, list), response
                        assert isinstance(alias, str), response
                        duplicates.append(target)
                        duplicates = list(set(duplicates))
                        assert all(
                            e in candidates or e == target for e in duplicates
                        ), f"{target}\n{duplicates}\n{candidates}"
                if self.verbose:
                    print(f"Entity aliases for '{alias}': {duplicates}")
                if alias in entity_aliases:
                    for f in duplicates:
                        if f not in entity_aliases[alias]:
                            entity_aliases[alias].append(f)
                else:
                    entity_aliases[alias] = duplicates
                for e in duplicates:
                    if e in remaining:
                        remaining.remove(e)
        return entity_aliases, usage_info
