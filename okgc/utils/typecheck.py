from collections import defaultdict

from okgc.utils.graph import Graph
from okgc.utils.sparql import (
    Predicate,
    TypeInfo,
    TypesHierarchy,
)


def check_valid_types(
    types: list[TypeInfo], allowed_types: list[TypeInfo], *, hierarchy: TypesHierarchy
) -> bool:
    assert len(types) > 0
    # If the list of allowed types is empty, we assume that any type is valid
    # This happens for example when a predicate on Wikidata does not have any
    # domain/range information, e.g., the predicate "has part(s)"
    if not allowed_types:
        return True
    for t1 in types:
        for t2 in allowed_types:
            # If the cose is the same, then we have a match
            if hierarchy.is_subclass(t1, t2):
                return True
    # No match has been found
    return False


def satisfy_domain_range_constraint(
    predicate: Predicate,
    subject_types: list[TypeInfo],
    object_types: list[TypeInfo],
    *,
    hierarchy: TypesHierarchy,
) -> tuple[bool, bool]:
    if predicate.is_unknown():
        return False, False
    ok_subject_types = check_valid_types(
        subject_types,
        [hierarchy[c] for c in predicate.domain],
        hierarchy=hierarchy,
    )
    ok_object_types = check_valid_types(
        object_types,
        [hierarchy[c] for c in predicate.range],
        hierarchy=hierarchy,
    )
    return ok_subject_types, ok_object_types


def satisfy_qualifier_constraints(
    predicate: Predicate,
    qualifier: Predicate,
    object_types: list[TypeInfo],
    *,
    hierarchy: TypesHierarchy,
) -> tuple[bool, bool]:
    ok_predicate = not predicate.is_unknown()
    ok_predicate &= (
        predicate.qualifiers is None or qualifier.code in predicate.qualifiers
    )
    ok_object_types = not qualifier.is_unknown()
    ok_object_types &= check_valid_types(
        object_types,
        [hierarchy[c] for c in qualifier.range],
        hierarchy=hierarchy,
    )
    return ok_predicate, ok_object_types


def ontology_filter_triples(
    graph: Graph,
    *,
    hierarchy: TypesHierarchy,
    filter_qualifiers: bool = False,
) -> Graph:
    kg = Graph(
        entities=set(),
        entity_types={},
        entity_descriptions=None if graph.entity_descriptions is None else {},
        predicates={},
        triples=set(),
        entity_text_ids={},
        triple_text_ids={},
        qualifiers=None,
        entity_aliases=None if graph.entity_aliases is None else {},
    )
    for t in graph.triples:
        s, p_id, o = t
        ok_subject_type, ok_object_type = satisfy_domain_range_constraint(
            graph.predicates[p_id],
            subject_types=graph.entity_types[s],
            object_types=graph.entity_types[o],
            hierarchy=hierarchy,
        )
        if not ok_subject_type or not ok_object_type:
            continue
        kg.triples.add(t)
        kg.predicates[p_id] = graph.predicates[p_id]
        kg.triple_text_ids[t] = graph.triple_text_ids[t]
        if graph.qualifiers is None or t not in graph.qualifiers:
            continue
        if kg.qualifiers is None:
            kg.qualifiers = defaultdict(list)
        for q in graph.qualifiers[t]:
            q_id, q_o = q
            q_p = graph.predicates[q_id]
            if not filter_qualifiers:
                kg.qualifiers[t].append(q)
                kg.predicates[q_id] = q_p
                continue
            ok_qualifier, ok_object_type = satisfy_qualifier_constraints(
                graph.predicates[p_id],
                q_p,
                graph.entity_types[q_o],
                hierarchy=hierarchy,
            )
            if not ok_qualifier or not ok_object_type:
                continue
            kg.qualifiers[t].append(q)
            kg.predicates[q_id] = q_p

    kg.entities = set(s for s, _, _ in kg.triples) | set(o for _, _, o in kg.triples)
    if kg.qualifiers is not None:
        kg.entities = kg.entities | set(
            q_o for _, qs in kg.qualifiers.items() for _, q_o in qs
        )
    for e in kg.entities:
        if e in graph.entity_types:
            kg.entity_types[e] = graph.entity_types[e]
        if kg.entity_descriptions is not None and e in graph.entity_descriptions:
            kg.entity_descriptions[e] = graph.entity_descriptions[e]
        if e in graph.entity_text_ids:
            kg.entity_text_ids[e] = graph.entity_text_ids[e]
        if kg.entity_aliases is not None and e in graph.entity_aliases:
            kg.entity_aliases[e] = graph.entity_aliases[e]
    return kg


def remove_qualifiers(graph: Graph) -> Graph:
    kg = Graph(
        entities=set(),
        entity_types={},
        entity_descriptions=None if graph.entity_descriptions is None else {},
        predicates={},
        triples=graph.triples,
        entity_text_ids={},
        triple_text_ids=graph.triple_text_ids,
        qualifiers=None,
        entity_aliases=None if graph.entity_aliases is None else {},
    )
    for _, r, _ in graph.triples:
        if r not in kg.predicates and r in graph.predicates:
            kg.predicates[r] = graph.predicates[r]
    kg.entities = set(s for s, _, _ in kg.triples) | set(o for _, _, o in kg.triples)
    for e in kg.entities:
        if e in graph.entity_types:
            kg.entity_types[e] = graph.entity_types[e]
        if kg.entity_descriptions is not None and e in graph.entity_descriptions:
            kg.entity_descriptions[e] = graph.entity_descriptions[e]
        if e in graph.entity_text_ids:
            kg.entity_text_ids[e] = graph.entity_text_ids[e]
        if kg.entity_aliases is not None and e in graph.entity_aliases:
            kg.entity_aliases[e] = graph.entity_aliases[e]
    return kg


def is_swap_allowed(
    predicate: Predicate,
    subject_types: list[TypeInfo],
    object_types: list[TypeInfo],
    *,
    hierarchy: TypesHierarchy,
) -> bool:
    ok_subject_inv, ok_object_inv = satisfy_domain_range_constraint(
        predicate,
        object_types,
        subject_types,
        hierarchy=hierarchy,
    )
    return ok_subject_inv and ok_object_inv


def find_consistent_entities(
    entity_types: dict[str, list[TypeInfo]],
    allowed_types: list[str],
    *,
    hierarchy: TypesHierarchy,
) -> set[str]:
    consistent_entities: set[str] = set()
    for e, ts in entity_types.items():
        if not check_valid_types(
            ts,
            [hierarchy[c] for c in allowed_types],
            hierarchy=hierarchy,
        ):
            continue
        consistent_entities.add(e)
    return consistent_entities


def retrieve_predicate_direction(
    predicate: Predicate,
    subject_types: list[TypeInfo],
    object_types: list[TypeInfo],
    *,
    hierarchy: TypesHierarchy,
) -> bool | None:
    valid_fwd_subject, valid_fwd_object = satisfy_domain_range_constraint(
        predicate,
        subject_types,
        object_types,
        hierarchy=hierarchy,
    )
    if valid_fwd_subject and valid_fwd_object:
        return True
    valid_bwd_subject, valid_bwd_object = satisfy_domain_range_constraint(
        predicate,
        object_types,
        subject_types,
        hierarchy=hierarchy,
    )
    if valid_bwd_subject and valid_bwd_object:
        return False
    return None


def filter_consistent_predicates(
    candidate_predicates: list[Predicate],
    subject_types: list[TypeInfo],
    object_types: list[TypeInfo],
    *,
    hierarchy: TypesHierarchy,
    include_inverses: bool = False,
) -> list[Predicate]:
    consistent_predicates = []
    for p in candidate_predicates:
        ok_subject, ok_object = satisfy_domain_range_constraint(
            p,
            subject_types,
            object_types,
            hierarchy=hierarchy,
        )
        if ok_subject and ok_object:
            consistent_predicates.append(p)
            continue
        if not include_inverses:
            continue
        ok_subject_inv, ok_object_inv = satisfy_domain_range_constraint(
            p,
            object_types,
            subject_types,
            hierarchy=hierarchy,
        )
        if ok_subject_inv and ok_object_inv:
            consistent_predicates.append(p)
    return consistent_predicates


def filter_consistent_qualifiers(
    predicate: Predicate,
    object_types: list[TypeInfo],
    candidate_predicates: list[Predicate],
    *,
    hierarchy: TypesHierarchy,
) -> list[Predicate]:
    consistent_q_predicates = []
    for q_p in candidate_predicates:
        ok_predicate, ok_object_type = satisfy_qualifier_constraints(
            predicate,
            q_p,
            object_types,
            hierarchy=hierarchy,
        )
        if ok_predicate and ok_object_type:
            consistent_q_predicates.append(q_p)
    return consistent_q_predicates
