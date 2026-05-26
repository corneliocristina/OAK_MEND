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
from okgc.utils.usage import UsageInfo
from okgc.utils.vector_index import PredicatesVectorIndex, StrIndex, TypesVectorIndex


class InContextKGC:
    def __init__(
        self,
        client: OpenAIClient,
        sent_embed: SentenceEmbedder,
        *,
        types_index: TypesVectorIndex,
        predicates_index: PredicatesVectorIndex,
        qualifier_predicates_index: PredicatesVectorIndex,
        max_num_predicates: int = 20,
        max_num_qualifiers: int = 200,
        prompts_path: str | None = None,
        verbose: bool = False,
    ):
        assert max_num_predicates > 0
        assert max_num_qualifiers is None or max_num_qualifiers > 0
        if max_num_qualifiers is None:
            max_num_qualifiers = max_num_predicates
        if prompts_path is None:
            prompts_path = os.path.join("prompts", "in-context")
        self.client = client
        self.sent_embed = sent_embed
        self.types_index = types_index
        self.predicates_index = predicates_index
        self.qualifier_predicates_index = qualifier_predicates_index
        self.max_num_predicates = max_num_predicates
        self.max_num_qualifiers = max_num_qualifiers
        self.verbose = verbose
        self._nlp: spacy.Language | None = None

        self.prompts: dict[str, str] = {
            "extract_predicates": load_prompt(
                os.path.join(prompts_path, "extract_predicates_1shot.txt"),
            ),
            "extract_triples": load_prompt(
                os.path.join(prompts_path, "extract_triples_qualifiers_1shot.txt"),
            ),
            "extract_triples_fallback": load_prompt(
                os.path.join(
                    prompts_path, "extract_triples_qualifiers_1shot_fallback.txt"
                ),
            ),
            "deduplicate_entities": load_prompt(
                os.path.join(prompts_path, "deduplicate_entities.txt")
            ),
        }

    def generate_from_text(
        self, text: str, *, text_id: int = 0
    ) -> tuple[Graph, dict[str, UsageInfo]]:
        # Extract the raw predicates using LLMs and retrieve the K closest predicates in the ontology
        raw_predicates, ext_predicate_usage = self.extract_raw_predicates(text)
        found_predicates, scores = self.predicates_index.search(
            raw_predicates, k=self.max_num_predicates, return_scores=True
        )
        candidate_predicates: list[Predicate] = []
        for i in range(len(raw_predicates) * self.max_num_predicates):
            if len(candidate_predicates) >= self.max_num_predicates:
                break
            j, k = i % len(raw_predicates), i // len(raw_predicates)
            p = found_predicates[j][k]
            if any(p.code == u.code for u in candidate_predicates):
                continue
            candidate_predicates.append(p)

        # Obtain the allowed qualifiers from the candidate predicates
        allowed_qualifiers: dict[str, Predicate] = {}
        for p in candidate_predicates:
            if p.qualifiers is None:
                continue
            for c in p.qualifiers:
                if c in allowed_qualifiers:
                    continue
                allowed_qualifiers[c] = self.qualifier_predicates_index[c]
        candidate_qualifiers: list[Predicate] = list(allowed_qualifiers.values())
        candidate_qualifiers = candidate_qualifiers[: self.max_num_qualifiers]

        # Collect all the allowed entity types from the domain and range of the allowed predicates/qualifiers
        allowed_entity_type_codes = set()
        for p in candidate_predicates:
            for c in p.domain + p.range:
                allowed_entity_type_codes.add(c)
        for q in candidate_qualifiers:
            for c in q.range:
                allowed_entity_type_codes.add(c)
        candidate_types: list[TypeInfo] = [
            self.types_index[c] for c in allowed_entity_type_codes
        ]

        # Extract the triples by giving the predicates, the entity types, and the domain/range constraints
        # as part of the context in an LLM
        triples, predicates, entity_types, qualifiers, ext_triple_usage = (
            self.extract_triples(
                text, candidate_predicates, candidate_qualifiers, candidate_types
            )
        )
        entities = set(s for s, _, _ in triples) | set(o for _, _, o in triples)
        entities |= set(q_o for qs in qualifiers.values() for q_p_id, q_o in qs)
        entity_text_ids = {e: [text_id] for e in entities}
        triple_text_ids = {t: [text_id] for t in triples}

        out = Graph(
            entities=entities,
            entity_types=entity_types,
            entity_descriptions=None,
            predicates=predicates,
            triples=triples,
            entity_text_ids=entity_text_ids,
            triple_text_ids=triple_text_ids,
            qualifiers=qualifiers,
        )

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
            "extract_predicates": ext_predicate_usage,
            "extract_triples": ext_triple_usage,
        }

        return out, usage_info

    def extract_raw_predicates(self, text: str) -> tuple[list[str], UsageInfo]:
        # Call the LLM to extract the predicates
        user_prompt = f"""### Input\nText:\n{text}\n\n"""
        predicates, usage_info = self.client.get_completion(
            self.prompts["extract_predicates"],
            user_prompt,
            transform_to_json=True,
        )
        assert isinstance(predicates, list) and all(
            isinstance(r, str) for r in predicates
        ), predicates
        return predicates, usage_info

    def extract_triples(
        self,
        text: str,
        candidate_predicates: list[Predicate],
        candidate_qualifiers: list[Predicate],
        candidate_entity_types: list[TypeInfo],
    ) -> tuple[
        set[Triple],
        dict[str, Predicate],
        dict[str, list[TypeInfo]],
        dict[Triple, list[Qualifier]],
        UsageInfo,
    ]:
        # Prepare the fallback map of allowed entity types, using the "basenames" of the types
        # i.e., the name of a type without the description (used to disambiguate types with the same name)
        named_entity_types: dict[str, TypeInfo] = {
            t.name: t for t in candidate_entity_types
        }
        named_entity_types_fallback = named_entity_types.copy()
        named_entity_types_fallback.update(
            {t.basename(): t for c, t in named_entity_types.items()}
        )

        # Prepare the maps from predicate/qualifier names to predicates
        named_predicates: dict[str, Predicate] = {
            p.name: p for p in candidate_predicates
        }
        named_qualifiers: dict[str, Predicate] = {
            q.name: q for q in candidate_qualifiers
        }

        # Second, prepare the list of predicates and their constraints to put in the context
        filtered_candidate_qualifier_codes = []
        for p in candidate_predicates:
            if p.qualifiers is None:
                filtered_candidate_qualifier_codes.append([])
                continue
            qs = []
            for q_id in p.qualifiers:
                q = self.qualifier_predicates_index[q_id]
                # Skip those qualifiers that are not in our ranking of qualifiers that have been extracted
                if q.name not in named_qualifiers:
                    continue
                qs.append(q.code)
            filtered_candidate_qualifier_codes.append(qs)
        allowed_predicates_repr = "\n".join(
            f"- {p.name}\n"
            + f"    - Domain: {[self.types_index[c].name for c in p.domain]}\n"
            + f"    - Range: {[self.types_index[c].name for c in p.range]}\n"
            + f"    - Qualifiers: {[self.qualifier_predicates_index[c].name for c in qs]}"
            for p, qs in zip(candidate_predicates, filtered_candidate_qualifier_codes)
        )

        # Third, prepare the list of qualifiers
        allowed_qualifiers_repr = "\n".join(
            f"- {q.name}\n"
            + f"    - Range: {[self.types_index[c].name for c in q.range]}"
            for q in candidate_qualifiers
        )

        if self.verbose:
            print("Allowed predicates:", list(candidate_predicates))
            print("Allowed qualifiers:", list(candidate_qualifiers))
            print("Allowed types:", list(candidate_entity_types))

        # Call the LLM to extract the triples
        user_prompt = f"""### Input\nText:\n{text}\n\n"""
        user_prompt += f"""Allowed predicates:\n{allowed_predicates_repr}\n\n"""
        user_prompt += f"""Allowed qualifiers:\n{allowed_qualifiers_repr}\n"""
        raw_output, usage_info = self.client.get_completion(
            self.prompts["extract_triples"],
            user_prompt,
            transform_to_json=True,
        )
        assert isinstance(raw_output, list)

        try:
            parse_output = self.parse_raw_triples(
                raw_output,
                named_predicates,
                named_qualifiers,
                named_entity_types,
                named_entity_types_fallback,
            )
        except:
            # Get completion (fallback prompt)
            raw_output, usage_info = self.client.get_completion(
                self.prompts["extract_triples_fallback"],
                user_prompt,
                transform_to_json=True,
            )
            assert isinstance(raw_output, list)
            parse_output = self.parse_raw_triples(
                raw_output,
                named_predicates,
                named_qualifiers,
                named_entity_types,
                named_entity_types_fallback,
                fallback=True,
            )
        triples, predicates, entity_types, qualifiers = parse_output
        return triples, predicates, entity_types, qualifiers, usage_info

    def parse_raw_triples(
        self,
        raw_output: list[dict[str, Any]],
        named_predicates: dict[str, Predicate],
        named_qualifiers: dict[str, Predicate],
        named_entity_types: dict[str, TypeInfo],
        named_entity_types_fallback: dict[str, TypeInfo],
        *,
        fallback: bool = False,
    ) -> tuple[
        set[Triple],
        dict[str, Predicate],
        dict[str, list[TypeInfo]],
        dict[Triple, list[Qualifier]],
    ]:
        triples: set[Triple] = set()
        predicates: dict[str, Predicate] = {}
        entity_types: dict[str, list[TypeInfo]] = defaultdict(list)
        qualifiers: dict[Triple, list[Qualifier]] = defaultdict(list)

        for entry in raw_output:
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
            # Gather the predicates
            if r in named_predicates:
                p = named_predicates[r]
            else:
                p = Predicate.unknown_from_name(r)
            if p.code not in predicates:
                predicates[p.code] = p
            triple = (s, p.code, o)
            subject_type = entry["subject_type"]
            object_type = entry["object_type"]
            subject_type = str(subject_type)
            object_type = str(object_type)
            subject_type = normalize_lowercase(subject_type)
            object_type = normalize_lowercase(object_type)
            if not subject_type:
                subject_type = "None"
            if not object_type:
                object_type = "None"
            if fallback:
                qs = [
                    (q["predicate"], q["object"], q["object_type"])
                    for q in entry["qualifiers"]
                ]
            else:
                qs = [
                    (*q["pair"], q["object_type"]) for q in entry.get("qualifiers", [])
                ]

            # Gather the triples
            triples.add(triple)

            # Gather the entity types
            for entity, type_name in zip([s, o], [subject_type, object_type]):
                if type_name in named_entity_types:
                    t = named_entity_types[type_name]
                elif type_name in named_entity_types_fallback:
                    t = named_entity_types_fallback[type_name]
                else:
                    t = TypeInfo.unknown_from_name(type_name)
                if not any(t.code == u.code for u in entity_types[entity]):
                    entity_types[entity].append(t)

            # Gather the triple qualifiers
            for q_r, q_o, q_object_type in qs:
                q_r = normalize_lowercase(q_r)
                if q_r in named_qualifiers:
                    q_p = named_qualifiers[q_r]
                else:
                    q_p = Predicate.unknown_from_name(q_r)
                if q_p.code not in predicates:
                    predicates[q_p.code] = q_p
                q_object_type = str(q_object_type)
                q_object_type = normalize_lowercase(q_object_type)
                if not q_object_type:
                    q_object_type = "None"
                if q_object_type in named_entity_types:
                    t = named_entity_types[q_object_type]
                elif q_object_type in named_entity_types_fallback:
                    t = named_entity_types_fallback[q_object_type]
                else:
                    t = TypeInfo.unknown_from_name(q_object_type)
                if not any(t.code == u.code for u in entity_types[q_o]):
                    entity_types[q_o].append(t)
                q = (q_p.code, q_o)
                if q not in qualifiers[triple]:
                    qualifiers[triple].append(q)

        return triples, predicates, entity_types, qualifiers

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
