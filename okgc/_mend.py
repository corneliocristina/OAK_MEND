import os
from copy import deepcopy

from okgc.utils.filesystem import load_prompt
from okgc.utils.graph import (
    Graph,
    Qualifier,
    Triple,
    merge_entity_types,
)
from okgc.utils.openai_wrapper import OpenAIClient
from okgc.utils.sparql import (
    Predicate,
    TypeInfo,
)
from okgc.utils.transform import (
    Transform,
    TransformCounters,
    TransformKind,
)
from okgc.utils.typecheck import (
    filter_consistent_predicates,
    filter_consistent_qualifiers,
    is_swap_allowed,
    satisfy_domain_range_constraint,
    satisfy_qualifier_constraints,
)
from okgc.utils.usage import UsageInfo
from okgc.utils.vector_index import PredicatesVectorIndex, TypesVectorIndex


class OntologyMend:
    def __init__(
        self,
        client: OpenAIClient,
        *,
        correct_qualifiers: bool = False,
        types_index: TypesVectorIndex,
        predicates_index: PredicatesVectorIndex,
        qualifier_predicates_index: PredicatesVectorIndex,
        prompts_path: str | None = None,
        verbose: bool = False,
    ):
        if prompts_path is None:
            prompts_path = os.path.join("prompts", "oak-mend")
        self.client = client
        self.types_index = types_index
        self.predicates_index = predicates_index
        self.qualifier_predicates_index = qualifier_predicates_index
        self.correct_qualifiers = correct_qualifiers
        self.verbose = verbose

        self.prompts: dict[str, str] = {
            "correct_triple": load_prompt(
                os.path.join(prompts_path, "correct_triple.txt")
            ),
            "correct_qualifier": load_prompt(
                os.path.join(prompts_path, "correct_qualifier.txt")
            ),
        }
        self.triples_transform_counters = TransformCounters()
        self.qualifiers_transform_counters = TransformCounters()

    def reset(self):
        self.triples_transform_counters = TransformCounters()
        self.qualifiers_transform_counters = TransformCounters()

    def apply_repeated(
        self, graph: Graph, texts: dict[int, str], *, max_budget: int = 2
    ) -> tuple[Graph, dict[Triple, Triple], UsageInfo]:
        out = graph
        origins = {t: t for t in graph.triples}
        usage_info = UsageInfo()
        for _ in range(max_budget):
            out, cur_origins, cur_usage_info = self.apply(out, texts)
            origins = {t: origins[orig] for t, orig in cur_origins.items()}
            if cur_usage_info.is_zero():
                break
            usage_info += cur_usage_info
        return out, origins, usage_info

    def apply(
        self, graph: Graph, texts: dict[int, str]
    ) -> tuple[Graph, dict[Triple, Triple], UsageInfo]:
        out = Graph(
            entities=graph.entities.copy(),
            entity_types=deepcopy(graph.entity_types),
            entity_descriptions=None,
            predicates={},
            triples=set(),
            entity_text_ids=deepcopy(graph.entity_text_ids),
            triple_text_ids={},
            qualifiers=None,
            entity_aliases=graph.entity_aliases,
        )
        origins = {}
        usage_info = UsageInfo()

        # Go through each triple, one at a time
        for triple in graph.triples:
            # Get the text(s) where the triple has been extracted from
            text = "\n\n".join(
                texts[i] for i in sorted(set(graph.triple_text_ids[triple]))
            )

            # Check and optionally correct the triple and its qualifiers (if any)
            usage_info += self.apply_triple(
                triple, text, origins=origins, graph=graph, out=out
            )

        return out, origins, usage_info

    def apply_triple(
        self,
        triple: Triple,
        text: str,
        *,
        origins: dict[Triple, Triple],
        graph: Graph,
        out: Graph,
    ) -> UsageInfo:
        s, p_id, o = triple
        predicate = graph.predicates[p_id]
        subject_types = out.entity_types[s]
        object_types = out.entity_types[o]
        labeled_triple = (s, predicate.name, o)

        # If the predicate is unknown, i.e., it cannot be canonicalized without making
        # the triple inconsistent, then we retrieve the candidate canonical predicate first
        if predicate.is_unknown():
            assert predicate.candidate_code is not None
            canonical_predicate = self.predicates_index[predicate.candidate_code]
        else:
            canonical_predicate = predicate

        # (0) Check whether the triple satisfies domain and range constraints
        ok_subject, ok_object = satisfy_domain_range_constraint(
            canonical_predicate,
            subject_types,
            object_types,
            hierarchy=self.types_index.hierarchy,
        )
        # If the triple is consistent, then add it to the set of triples
        if ok_subject and ok_object:
            return self.add_triple(
                s,
                canonical_predicate,
                o,
                {},
                text,
                orig=triple,
                origins=origins,
                graph=graph,
                out=out,
                is_valid=True,
            )

        # Next, we predict the triple correction transformation
        #      (1) Try to swap subject and object and, if it fails,
        # then (2) Ask the LLM to correct the triple based on the relevant text
        transforms, usage_info = self.predict_triple_transforms(
            triple,
            text,
            predicate=predicate,
            canonical_predicate=canonical_predicate,
            subject_types=subject_types,
            object_types=object_types,
            ok_subject=ok_subject,
            ok_object=ok_object,
        )
        if self.verbose:
            print(f"{labeled_triple} -- Predicted: {transforms}", usage_info)
        # If no transforms have been found, add the triple to the graph as an invalid one
        if not transforms:
            usage_info += self.add_triple(
                s,
                predicate,
                o,
                {},
                text,
                orig=triple,
                origins=origins,
                graph=graph,
                out=out,
                is_valid=False,
            )
            return usage_info

        # Apply transforms and obtain candidate triples and candidate entity types
        candidate_triple, candidate_entity_types = self.apply_triple_transforms(
            triple, transforms, graph=graph
        )
        # Check whether the transformed triple is valid
        candidate_s, candidate_p_id, candidate_o = candidate_triple
        candidate_subject_types = out.entity_types[
            candidate_s
        ] + candidate_entity_types.get(candidate_s, [])
        candidate_object_types = out.entity_types[
            candidate_o
        ] + candidate_entity_types.get(candidate_o, [])
        candidate_predicate = self.predicates_index[candidate_p_id]
        ok_subject, ok_object = satisfy_domain_range_constraint(
            candidate_predicate,
            candidate_subject_types,
            candidate_object_types,
            hierarchy=self.types_index.hierarchy,
        )
        orig = triple
        if not ok_subject or not ok_object:
            if predicate.is_unknown():
                candidate_predicate = Predicate.unknown_from_name(
                    predicate.name,
                    candidate_code=candidate_predicate.code,
                    description=predicate.description,
                )
            usage_info += self.add_triple(
                candidate_s,
                candidate_predicate,
                candidate_o,
                candidate_entity_types,
                text,
                orig=orig,
                origins=origins,
                graph=graph,
                out=out,
                is_valid=False,
            )
            return usage_info

        # The transform was successful, thus add the triple
        if self.verbose:
            print(f"{labeled_triple} -- {transforms}")
        usage_info += self.add_triple(
            candidate_s,
            candidate_predicate,
            candidate_o,
            candidate_entity_types,
            text,
            orig=orig,
            origins=origins,
            graph=graph,
            out=out,
            is_valid=True,
        )

        return usage_info

    def add_triple(
        self,
        s: str,
        p: Predicate,
        o: str,
        entity_types: dict[str, list[TypeInfo]],
        text: str,
        *,
        orig: Triple,
        origins: dict[Triple, Triple],
        graph: Graph,
        out: Graph,
        is_valid: bool = True,
    ) -> UsageInfo:
        triple = (s, p.code, o)
        if self.verbose:
            labeled_triple = (s, p.name, o)
            if is_valid:
                print(f"{labeled_triple} -- Adding valid triple")
            else:
                print(f"{labeled_triple} -- Keep invalid triple")
        origins[triple] = orig
        out.triples.add(triple)
        merge_entity_types(out.entity_types, entity_types, in_place=True)
        out.predicates[p.code] = p
        out.triple_text_ids[triple] = graph.triple_text_ids[orig]
        if is_valid:
            return self.apply_triple_qualifiers(
                triple, orig, text, graph=graph, out=out
            )
        usage_info = UsageInfo()
        if graph.qualifiers is None or orig not in graph.qualifiers:
            return usage_info
        if out.qualifiers is None:
            out.qualifiers = {}
        if triple not in out.qualifiers:
            out.qualifiers[triple] = []
        for q in graph.qualifiers[orig]:
            q_id, _ = q
            out.predicates[q_id] = graph.predicates[q_id]
            out.qualifiers[triple].append(q)
        return usage_info

    def add_qualifier(
        self,
        triple: Triple,
        q_predicate: Predicate,
        q_object: str,
        entity_types: dict[str, list[TypeInfo]],
        *,
        out: Graph,
        is_valid: bool = True,
    ):
        qualifier = (q_predicate.code, q_object)
        if self.verbose:
            labeled_qualifier = (q_predicate.name, q_object)
            if is_valid:
                print(f"     :: {labeled_qualifier} -- Adding valid qualifier")
            else:
                print(f"     :: {labeled_qualifier} -- Keeping invalid qualifier")
        assert out.qualifiers is not None
        assert triple in out.qualifiers
        out.qualifiers[triple].append(qualifier)
        out.predicates[q_predicate.code] = q_predicate
        merge_entity_types(out.entity_types, entity_types, in_place=True)

    def apply_triple_qualifiers(
        self,
        triple: Triple,
        orig: Triple,
        text: str,
        *,
        graph: Graph,
        out: Graph,
    ) -> UsageInfo:
        if graph.qualifiers is None or orig not in graph.qualifiers:
            return UsageInfo()
        if out.qualifiers is None:
            out.qualifiers = {}
        if triple not in out.qualifiers:
            out.qualifiers[triple] = []

        usage_info = UsageInfo()
        for q in graph.qualifiers[orig]:
            usage_info += self.apply_qualifier(triple, q, text, graph=graph, out=out)
        return usage_info

    def apply_qualifier(
        self,
        triple: Triple,
        qualifier: Qualifier,
        text: str,
        *,
        graph: Graph,
        out: Graph,
    ) -> UsageInfo:
        s, p_id, o = triple
        predicate = out.predicates[p_id]
        if self.verbose:
            labeled_triple = (s, predicate.name, o)
            print(f"{labeled_triple}")
        q_id, q_o = qualifier
        q_p = graph.predicates[q_id]
        object_types = out.entity_types[q_o]
        labeled_qualifier = (q_p.name, q_o)

        # If the qualifier predicate is unknown, i.e., it cannot be canonicalized without making
        # the qualifier inconsistent w.r.t. the triple, then we retrieve the candidate canonical predicate first
        if q_p.is_unknown():
            assert q_p.candidate_code is not None
            canonical_q_p = self.qualifier_predicates_index[q_p.candidate_code]
        else:
            canonical_q_p = q_p

        # Check the qualifier is valid
        ok_predicate, ok_object_type = satisfy_qualifier_constraints(
            predicate,
            canonical_q_p,
            object_types,
            hierarchy=self.types_index.hierarchy,
        )
        if ok_predicate and ok_object_type:
            self.add_qualifier(triple, canonical_q_p, q_o, {}, out=out, is_valid=True)
            return UsageInfo()
        if not self.correct_qualifiers:
            self.add_qualifier(triple, q_p, q_o, {}, out=out, is_valid=False)
            return UsageInfo()

        # Ask the LLM to correct the qualifier
        transform, usage_info = self.llm_correct_qualifier(
            triple,
            q_p,
            q_o,
            canonical_q_p,
            text,
            ok_predicate=ok_predicate,
            ok_object_type=ok_object_type,
            out=out,
        )
        if self.verbose:
            print(f"     :: {labeled_qualifier} -- Predicted: {transform}")
        if transform is None:
            self.add_qualifier(triple, q_p, q_o, {}, out=out, is_valid=False)
            return UsageInfo()

        # Apply the transform to the qualifier
        #
        # If the qualifier predicate we started from is an unknown predicate,
        # then the first transformation should be its replacement with the canonical one,
        # in order to make sure we get rid of the unknown qualifier predicate after the transform
        candidate_qualifier, candidate_entity_types = self.apply_qualifier_transform(
            (canonical_q_p.code if q_p.is_unknown() else q_p, q_o), transform
        )
        candidate_q_id, candidate_object = candidate_qualifier
        candidate_q_predicate = self.qualifier_predicates_index[candidate_q_id]
        assert candidate_object == q_o
        candidate_object_types = out.entity_types[q_o] + candidate_entity_types.get(
            q_o, []
        )
        ok_predicate, ok_object_type = satisfy_qualifier_constraints(
            predicate,
            candidate_q_predicate,
            candidate_object_types,
            hierarchy=self.types_index.hierarchy,
        )
        if not ok_predicate or not ok_object_type:
            if q_p.is_unknown():
                candidate_q_predicate = Predicate.unknown_from_name(
                    q_p.name,
                    candidate_code=candidate_q_predicate.code,
                    description=q_p.description,
                )
            self.add_qualifier(
                triple,
                candidate_q_predicate,
                candidate_object,
                candidate_entity_types,
                out=out,
                is_valid=False,
            )
            return usage_info
        if self.verbose:
            print(f"     :: {labeled_qualifier} -- {transform}")
        self.add_qualifier(
            triple,
            candidate_q_predicate,
            candidate_object,
            candidate_entity_types,
            out=out,
            is_valid=True,
        )
        return usage_info

    def predict_triple_transforms(
        self,
        triple: Triple,
        text: str,
        *,
        predicate: Predicate,
        canonical_predicate: Predicate,
        subject_types: list[TypeInfo],
        object_types: list[TypeInfo],
        ok_subject: bool,
        ok_object: bool,
    ) -> tuple[list[Transform], UsageInfo]:
        assert not ok_subject or not ok_object

        # (2) Try to swap subject and object
        ok_swap = is_swap_allowed(
            canonical_predicate,
            subject_types,
            object_types,
            hierarchy=self.types_index.hierarchy,
        )
        if ok_swap:
            transforms = [Transform(TransformKind.SWAP_SUBJECT_OBJECT)]
            usage_info = UsageInfo()
        else:
            # Otherwise, (3) Ask the LLM to correct the triple
            transforms, usage_info = self.llm_correct_triple(
                triple,
                text,
                predicate=predicate,
                canonical_predicate=canonical_predicate,
                subject_types=subject_types,
                object_types=object_types,
                ok_subject=ok_subject,
                ok_object=ok_object,
            )
        # If the predicate we started from is an unknown predicate,
        # then the first transformation should be its replacement with the canonical one,
        # in order to make sure we get rid of the unknown predicate after the transforms composition
        if (
            transforms
            and predicate.is_unknown()
            and not any(
                transform.kind == TransformKind.REPLACE_PREDICATE
                for transform in transforms
            )
        ):
            transforms.insert(
                0,
                Transform(
                    TransformKind.REPLACE_PREDICATE, predicate=canonical_predicate
                ),
            )
        return transforms, usage_info

    def apply_triple_transforms(
        self,
        triple: Triple,
        transforms: list[Transform],
        *,
        graph: Graph,
    ) -> tuple[Triple, dict[str, list[TypeInfo]]]:
        # Apply the transforms to a single triple in sequence,
        # and return the updated triple as well as new entity types (if any)
        s, p_id, o = triple
        entity_types = {}
        predicate = graph.predicates[p_id]
        for transform in transforms:
            if transform.kind == TransformKind.SWAP_SUBJECT_OBJECT:
                triple = (o, predicate.code, s)
                s, o = o, s
                self.triples_transform_counters.swap_subject_object += 1
            elif transform.kind == TransformKind.ADD_SUBJECT_TYPE:
                assert transform.entity_type is not None
                merge_entity_types(
                    entity_types, {s: [transform.entity_type]}, in_place=True
                )
                self.triples_transform_counters.add_subject_type += 1
            elif transform.kind == TransformKind.ADD_OBJECT_TYPE:
                assert transform.entity_type is not None
                merge_entity_types(
                    entity_types, {o: [transform.entity_type]}, in_place=True
                )
                self.triples_transform_counters.add_object_type += 1
            elif transform.kind == TransformKind.REPLACE_PREDICATE:
                predicate = transform.predicate
                assert predicate is not None
                triple = (s, predicate.code, o)
                self.triples_transform_counters.replace_predicate += 1
            else:
                raise ValueError(f"Unknown transform kind '{transform.kind}'")
        return triple, entity_types

    def apply_qualifier_transform(
        self, qualifier: Qualifier, transform: Transform
    ) -> tuple[Qualifier, dict[str, list[TypeInfo]]]:
        q_id, q_o = qualifier
        # Apply the transform to the qualifier
        if transform.kind == TransformKind.ADD_OBJECT_TYPE:
            assert transform.entity_type is not None
            entity_types = {q_o: [transform.entity_type]}
            self.qualifiers_transform_counters.add_object_type += 1
        elif transform.kind == TransformKind.REPLACE_PREDICATE:
            assert transform.predicate is not None
            q_id = transform.predicate.code
            entity_types = {}
            self.qualifiers_transform_counters.replace_predicate += 1
        else:
            raise ValueError(f"Unknown transform kind '{transform.kind}'")
        return (q_id, q_o), entity_types

    def llm_correct_triple(
        self,
        triple: Triple,
        text: str,
        *,
        predicate: Predicate,
        canonical_predicate: Predicate,
        subject_types: list[TypeInfo],
        object_types: list[TypeInfo],
        ok_subject: bool,
        ok_object: bool,
        max_retrieved_predicates: int = 250,
        max_candidate_predicates: int = 10,
    ) -> tuple[list[Transform], UsageInfo]:
        assert max_candidate_predicates < max_retrieved_predicates
        assert not ok_subject or not ok_object
        s, p_id, o = triple

        # Search for the closest predicates by embedding similarity
        retrieved_predicates = self.predicates_index.search(
            predicate.name, k=max_retrieved_predicates
        )
        assert isinstance(retrieved_predicates, list)
        retrieved_predicates = retrieved_predicates[0]

        # Obtain candidate predicates from the retrieved predicates,
        # i.e., candidates that are consistent with the types of subject and object,
        # or alternatively candidates that are consistent with the types of object and subject (inverted)
        candidate_predicates = filter_consistent_predicates(
            retrieved_predicates,
            subject_types,
            object_types,
            hierarchy=self.types_index.hierarchy,
            include_inverses=True,
        )
        candidate_predicates = candidate_predicates[:max_candidate_predicates]
        candidate_named_predicates = {p.name: p for p in candidate_predicates}

        # Collect the types in the domain and range of the candidate canonical predicate
        predicate_domain_named_types = {
            self.types_index[c].name: self.types_index[c]
            for c in canonical_predicate.domain
        }
        predicate_range_named_types = {
            self.types_index[c].name: self.types_index[c]
            for c in canonical_predicate.range
        }
        predicate_domain_named_types_fallback = {
            self.types_index[c].basename(): self.types_index[c]
            for c in canonical_predicate.domain
        }
        predicate_range_named_types_fallback = {
            self.types_index[c].basename(): self.types_index[c]
            for c in canonical_predicate.range
        }
        named_types = predicate_domain_named_types.copy()
        named_types.update(predicate_range_named_types)
        named_types_fallback = predicate_domain_named_types_fallback.copy()
        named_types_fallback.update(predicate_range_named_types_fallback)

        # Construct the prompt
        # Inputs:
        # 0. The source text(s) where the triple has been extracted from
        # 1. The triple having the closest predicate in the ontology (based on embedding similarity)
        # 2. A textual description on why the triple is not valid
        # 3. The types of the subject in the triple
        # 4. The types of the object in the triple
        # 5. The domain of the closest predicate in the ontology
        # 6. The range of the closest predicate in the ontology
        # 7. Candidate predicates from the ontology that are consistent with the subject and object types
        # Outputs:
        # A list of pairs each of the form
        # ["<Action>", "<Argument>"]
        # E.g., [["replace_predicate", "educated_at"]]
        triple_repr = f"{(s, canonical_predicate.name, o)}"
        predicate_domain_repr = "\n".join(
            f"- {name}" for name in predicate_domain_named_types.keys()
        )
        predicate_range_repr = "\n".join(
            f"- {name}" for name in predicate_range_named_types.keys()
        )
        candidate_predicates_repr = "\n".join(
            f"- {p.name}" for p in candidate_predicates
        )
        user_prompt = f"""### Input\nText:\n{text}\n\n"""
        user_prompt += f"""Triple: {triple_repr}\n\n"""
        user_prompt += f"""Domain of predicate {repr(canonical_predicate.name)}:\n{predicate_domain_repr}\n\n"""
        user_prompt += f"""Range of predicate {repr(canonical_predicate.name)}:\n{predicate_range_repr}\n\n"""
        user_prompt += f"""Candidate predicates:\n{candidate_predicates_repr}\n\n"""
        invalid_reasons = ""
        if not ok_subject:
            invalid_reasons += (
                f"- The entity {repr(s)} has types {[t.name for t in subject_types]}, "
            )
            invalid_reasons += f"but {repr(canonical_predicate.name)} requires {repr(s)} to be consistent with some type in the domain"
        if not ok_object:
            if not ok_subject:
                invalid_reasons += "\n"
            invalid_reasons += (
                f"- The entity {repr(o)} has types {[t.name for t in object_types]}, "
            )
            invalid_reasons += f"but {repr(canonical_predicate.name)} requires {repr(o)} to be consistent with some type in the range"
        user_prompt += f"""Reasons:\n{invalid_reasons}\n"""
        # if self.verbose:
        #    print("LLM triple correction prompt:")
        #    print(user_prompt)
        # Ask the LLM and parse the response
        response, usage_info = self.client.get_completion(
            self.prompts["correct_triple"],
            user_prompt=user_prompt,
            transform_to_json=True,
        )
        assert isinstance(response, list), response
        if len(response) == 1 and not response[0]:
            return [], usage_info
        assert all(isinstance(entry, (tuple, list)) for entry in response), response
        assert all(len(entry) == 1 or len(entry) == 2 for entry in response), response
        response = [
            (entry[0], None) if len(entry) == 1 else entry for entry in response
        ]
        transforms: list[Transform] = []
        for action, argument in response:
            if action == "swap_subject_object":
                transform = Transform(TransformKind.SWAP_SUBJECT_OBJECT)
            elif action == "add_subject_type":
                if argument in named_types:
                    entity_type = named_types[argument]
                elif argument in named_types_fallback:
                    entity_type = named_types_fallback[argument]
                else:
                    return [], usage_info
                transform = Transform(
                    TransformKind.ADD_SUBJECT_TYPE,
                    entity_type=entity_type,
                )
            elif action == "add_object_type":
                if argument in named_types:
                    entity_type = named_types[argument]
                elif argument in named_types_fallback:
                    entity_type = named_types_fallback[argument]
                else:
                    return [], usage_info
                transform = Transform(
                    TransformKind.ADD_OBJECT_TYPE,
                    entity_type=entity_type,
                )
            elif action == "replace_predicate":
                if argument not in candidate_named_predicates:
                    return [], usage_info
                p = candidate_named_predicates[argument]
                transform = Transform(TransformKind.REPLACE_PREDICATE, predicate=p)
            else:
                if self.verbose:
                    print(f"Unknown action predicted: '{action}'")
                return [], usage_info
            transforms.append(transform)
        return transforms, usage_info

    def llm_correct_qualifier(
        self,
        triple: Triple,
        q_predicate: Predicate,
        q_object: str,
        canonical_q_predicate: Predicate,
        text: str,
        *,
        ok_predicate: bool,
        ok_object_type: bool,
        out: Graph,
        max_retrieved_predicates: int = 250,
        max_candidate_q_predicates: int = 10,
    ) -> tuple[Transform | None, UsageInfo]:
        s, p_id, o = triple
        predicate = out.predicates[p_id]
        assert not predicate.is_unknown()
        q_object_types = out.entity_types[q_object]

        # Search the most similar predicates via embedding similarity
        candidate_q_predicates = self.qualifier_predicates_index.search(
            q_predicate.name, k=max_retrieved_predicates
        )
        assert isinstance(candidate_q_predicates, list)
        candidate_q_predicates = candidate_q_predicates[0]
        candidate_q_predicates = filter_consistent_qualifiers(
            predicate,
            q_object_types,
            candidate_q_predicates,
            hierarchy=self.types_index.hierarchy,
        )
        if predicate.qualifiers is not None:
            candidate_q_predicates = [
                q for q in candidate_q_predicates if q.code in predicate.qualifiers
            ]
        else:
            candidate_q_predicates = candidate_q_predicates[:max_candidate_q_predicates]
        if not candidate_q_predicates and not canonical_q_predicate.range:
            return None, UsageInfo()
        candidate_q_predicates_names = [p.name for p in candidate_q_predicates]
        candidate_q_predicates_map = dict(
            zip(candidate_q_predicates_names, candidate_q_predicates)
        )

        # Obtain the range types of the canonical qualifier predicate
        q_predicate_range_named_types = {
            self.types_index[c].name: self.types_index[c]
            for c in canonical_q_predicate.range
        }
        q_predicate_range_named_types_fallback = {
            self.types_index[c].basename(): self.types_index[c]
            for c in canonical_q_predicate.range
        }
        # Construct the prompt
        # Inputs:
        # 0. The source text(s) where the triple has been extracted from
        # 1. The triple of which the qualifier is connected to
        # 2. The invalid qualifier to correct
        # 3. A textual description on why the qualifier is not valid
        # 4. The types of the qualifier object
        # 5. The range types of the qualifier predicate
        # 6. Candidate qualifier predicates from the ontology that are consistent
        #    with both (i) the predicate qualifiers constraints and (ii) the qualifier object types
        # Outputs:
        # A dictionary of the form
        # {"action": <Action>, "argument": <Action Argument>}
        # E.g., {"action": "replace_predicate", "argument": "point in time"}
        triple_repr = f"{(s, predicate.name, o)}"
        qualifier_repr = f"{(canonical_q_predicate.name, q_object)}"
        qualifier_range_repr = "\n".join(
            f"- {name}" for name in q_predicate_range_named_types.keys()
        )
        candidate_qualifier_predicates_repr = "\n".join(
            f"- {name}" for name in candidate_q_predicates_names
        )
        if not ok_predicate:
            invalid_reason = f"- The predicate {repr(canonical_q_predicate.name)} is not allowed as qualifier by the predicate {repr(predicate.name)}"
        else:
            assert not ok_object_type
            invalid_reason = f"- The entity {repr(q_object)} has types {[t.name for t in q_object_types]}, "
            invalid_reason += f"but {repr(canonical_q_predicate.name)} requires {repr(q_object)} to be consistent with some type in the range"
        user_prompt = f"""### Input\nText:\n{text}\n\n"""
        user_prompt += f"""Triple: {triple_repr}\n\n"""
        user_prompt += f"""Qualifier: {qualifier_repr}\n\n"""
        user_prompt += f"""Range of predicate {repr(canonical_q_predicate.name)}:\n{qualifier_range_repr}\n\n"""
        user_prompt += f"""Candidate qualifier predicates:\n{candidate_qualifier_predicates_repr}\n\n"""
        user_prompt += f"""Reason:\n{invalid_reason}\n"""
        if self.verbose:
            print("LLM qualifier correction prompt:")
            print(user_prompt)

        # Ask the LLM and parse the response
        response, usage_info = self.client.get_completion(
            self.prompts["correct_qualifier"],
            user_prompt=user_prompt,
            transform_to_json=True,
        )
        if not response or response == "None":
            return None, usage_info
        if not isinstance(response, list) or len(response) != 2:
            return None, usage_info
        action, argument = response
        if action == "add_object_type":
            if argument in q_predicate_range_named_types:
                entity_type = q_predicate_range_named_types[argument]
            elif argument in q_predicate_range_named_types_fallback:
                entity_type = q_predicate_range_named_types_fallback[argument]
            else:
                return None, usage_info
            transform = Transform(
                TransformKind.ADD_OBJECT_TYPE,
                entity_type=entity_type,
            )
        elif action == "replace_predicate":
            if argument not in candidate_q_predicates_map:
                return None, usage_info
            q_predicate = candidate_q_predicates_map[argument]
            transform = Transform(
                TransformKind.REPLACE_PREDICATE, predicate=q_predicate
            )
        else:
            if self.verbose:
                print(f"Unknown action predicted: '{action}'")
            return None, usage_info
        return transform, usage_info
