from collections import defaultdict
from copy import deepcopy
from dataclasses import asdict, dataclass
from typing import Any

from dacite import Config, from_dict

from okgc.utils.sparql import Predicate, PredicateKind, TypeInfo

Triple = tuple[str, str, str]

Qualifier = tuple[str, str]


@dataclass
class Graph:
    entities: set[str]
    entity_types: dict[str, list[TypeInfo]]
    entity_descriptions: dict[str, str] | None
    predicates: dict[str, Predicate]
    triples: set[Triple]
    entity_text_ids: dict[str, list[int]]
    triple_text_ids: dict[Triple, list[int]]
    qualifiers: dict[Triple, list[Qualifier]] | None = None
    entity_aliases: dict[str, list[str]] | None = None


def graph_to_dict(g: Graph) -> dict[str, Any]:
    return {
        "entities": list(g.entities),
        "entity_types": {
            e: [asdict(t) for t in ts] for e, ts in g.entity_types.items()
        },
        "entity_descriptions": g.entity_descriptions,
        "predicates": {p: asdict(q) for p, q in g.predicates.items()},
        "triples": list(g.triples),
        "entity_text_ids": g.entity_text_ids,
        "triple_text_ids": [
            {"triple": t, "text_ids": ids} for t, ids in g.triple_text_ids.items()
        ],
        "qualifiers": [
            {"triple": t, "qualifiers": qs} for t, qs in g.qualifiers.items() if qs
        ]
        if g.qualifiers is not None
        else None,
        "entity_aliases": g.entity_aliases,
    }


def dict_to_graph(g: dict[str, Any]) -> Graph:
    return Graph(
        entities=set(g["entities"]),
        entity_types={
            e: [from_dict(TypeInfo, data=t) for t in ts]
            for e, ts in g["entity_types"].items()
        },
        entity_descriptions=g["entity_descriptions"],
        predicates={
            p: from_dict(Predicate, data=q, config=Config(cast=[PredicateKind]))
            for p, q in g["predicates"].items()
        },
        triples=set(Triple(t) for t in g["triples"]),
        entity_text_ids=g["entity_text_ids"],
        triple_text_ids={
            Triple(x["triple"]): x["text_ids"] for x in g["triple_text_ids"]
        },
        qualifiers={
            Triple(x["triple"]): [Qualifier(q) for q in x["qualifiers"]]
            for x in g["qualifiers"]
        }
        if g["qualifiers"] is not None
        else None,
        entity_aliases=g["entity_aliases"],
    )


def prune_unknown_types(
    entity_types: dict[str, list[TypeInfo]],
    in_place: bool = False,
) -> dict[str, list[TypeInfo]]:
    if not in_place:
        entity_types = deepcopy(entity_types)
    for e in list(entity_types.keys()):
        assert len(entity_types[e]) > 0
        filtered_types = [t for t in entity_types[e] if not t.is_unknown()]
        if not filtered_types:
            continue
        entity_types[e] = filtered_types
    return entity_types


def merge_entity_types(
    entity_types: dict[str, list[TypeInfo]],
    other: dict[str, list[TypeInfo]],
    in_place: bool = False,
    prune_useless_unknown: bool = True,
) -> dict[str, list[TypeInfo]]:
    if not in_place:
        entity_types = deepcopy(entity_types)
    for e, ts in other.items():
        for t in ts:
            if e not in entity_types:
                entity_types[e] = [t]
                continue
            if any(t.code == u.code for u in entity_types[e]):
                continue
            entity_types[e].append(t)
    if prune_useless_unknown:
        return prune_unknown_types(entity_types, in_place=True)
    return entity_types


def update_triple_text_ids(
    triple_text_ids: dict[Triple, list[int]],
    old_triple: Triple,
    new_triple: Triple,
    *,
    output: dict[Triple, list[int]],
):
    for text_id in triple_text_ids[old_triple]:
        if new_triple not in output:
            output[new_triple] = [text_id]
            continue
        if text_id not in output[new_triple]:
            output[new_triple].append(text_id)


def add_triple_qualifiers(
    triple: Triple,
    qualifiers: list[Qualifier],
    *,
    output: dict[Triple, list[Qualifier]],
):
    for q in qualifiers:
        if triple not in output:
            output[triple] = [q]
            continue
        if q not in output[triple]:
            output[triple].append(q)


def rename_entities(graph: Graph, *, index: int) -> Graph:
    entities_map: dict[str, str] = {e: f"{e} [{index}]" for e in graph.entities}
    triples_map: dict[Triple, Triple] = {
        (s, r, o): (entities_map[s], r, entities_map[o]) for s, r, o in graph.triples
    }
    entities = set(entities_map.values())
    triples = set(triples_map.values())
    entity_types = {entities_map[e]: ts for e, ts in graph.entity_types.items()}
    entity_descriptions = (
        None
        if graph.entity_descriptions is None
        else {entities_map[e]: desc for e, desc in graph.entity_descriptions.items()}
    )
    entity_text_ids = {
        entities_map[e]: text_ids for e, text_ids in graph.entity_text_ids.items()
    }
    triple_text_ids = {
        triples_map[t]: text_ids for t, text_ids in graph.triple_text_ids.items()
    }
    qualifiers = (
        {
            triples_map[t]: [(q_id, entities_map.get(q_o, q_o)) for q_id, q_o in qs]
            for t, qs in graph.qualifiers.items()
        }
        if graph.qualifiers is not None
        else None
    )
    entity_aliases = (
        {entities_map[e]: graph.entity_aliases.get(e, [e]) for e in graph.entities}
        if graph.entity_aliases is not None
        else None
    )
    out = Graph(
        entities,
        entity_types=entity_types,
        entity_descriptions=entity_descriptions,
        predicates=graph.predicates,
        triples=triples,
        entity_text_ids=entity_text_ids,
        triple_text_ids=triple_text_ids,
        qualifiers=qualifiers,
        entity_aliases=entity_aliases,
    )
    return out


def merge_graphs(graphs: list[Graph], *, duplicate_entities: bool = False) -> Graph:
    # If we want to merge the KGs but NOT merge the entities, we firstly rename the entities and then merge the KGs
    if duplicate_entities:
        return merge_graphs(
            [rename_entities(kg, index=i) for i, kg in enumerate(graphs)],
            duplicate_entities=False,
        )
    # Merge the generated knowledge graphs into a single one
    # To resolve types, at the moment we take the union of the types of colliding entities
    entities: set[str] = set()
    entity_types: dict[str, list[TypeInfo]] = defaultdict(list)
    entity_descriptions: dict[str, str] | None = None
    predicates: dict[str, Predicate] = {}
    triples: set[Triple] = set()
    qualifiers: dict[Triple, list[Qualifier]] | None = None
    entity_aliases: dict[str, list[str]] | None = None
    for g in graphs:
        # Merge the entities and the triples
        entities |= g.entities
        triples |= g.triples
        # Merge the entity types (and remove duplicates, if any)
        for e, ts in g.entity_types.items():
            for t in ts:
                if any(t.code == u.code for u in entity_types[e]):
                    continue
                entity_types[e].append(t)
        # Merge the entity descriptions, if any
        if g.entity_descriptions is not None:
            if entity_descriptions is None:
                entity_descriptions = {}
            for e in g.entities:
                if e in entity_descriptions:
                    entity_descriptions[e] += f" {g.entity_descriptions[e]}"
                else:
                    entity_descriptions[e] = g.entity_descriptions[e]
        # Merge the predicates
        for r, p in g.predicates.items():
            if r in predicates:
                continue
            predicates[r] = p
        # Merge the qualifiers, if any
        if g.qualifiers is not None:
            if qualifiers is None:
                qualifiers = defaultdict(list)
            for t, qs in g.qualifiers.items():
                for q in qs:
                    if q in qualifiers[t]:
                        continue
                    qualifiers[t].append(q)
        # Merge entity aliases, if any
        if g.entity_aliases is not None:
            if entity_aliases is None:
                entity_aliases = defaultdict(list)
            for e, als in g.entity_aliases.items():
                for a in als:
                    if a in entity_aliases[e]:
                        continue
                    entity_aliases[e].append(a)

    # Prune unknown entity types if they have been resolved after merging
    entity_types = prune_unknown_types(entity_types)

    # Collect the entities to text ids
    entity_text_ids: dict[str, list[int]] = {}
    for e in entities:
        text_ids = []
        for g in graphs:
            if e not in g.entity_text_ids:
                continue
            text_ids.extend(g.entity_text_ids[e])
        entity_text_ids[e] = list(set(text_ids))

    # Collect the triple to text ids
    triple_text_ids: dict[Triple, list[int]] = {}
    for t in triples:
        text_ids = []
        for g in graphs:
            if t not in g.triple_text_ids:
                continue
            text_ids.extend(g.triple_text_ids[t])
        triple_text_ids[t] = list(set(text_ids))

    # Construct the merged knowledge graph
    graph = Graph(
        entities,
        entity_types,
        entity_descriptions,
        predicates,
        triples,
        entity_text_ids=entity_text_ids,
        triple_text_ids=triple_text_ids,
        qualifiers=qualifiers,
        entity_aliases=entity_aliases,
    )
    return graph


def deduplicate_entities(graph: Graph, *, aliases: dict[str, list[str]]) -> Graph:
    # Collect the information of the clustered entities,
    # including their description, their types and the map to the text ids
    entities: set[str] = set(aliases.keys())
    entity_types: dict[str, list[TypeInfo]] = defaultdict(list)
    entity_descriptions: dict[str, str] | None = None
    if graph.entity_descriptions is not None:
        entity_descriptions = {}
    entity_text_ids: dict[str, list[int]] = defaultdict(list)
    for rep, cluster in aliases.items():
        for f in cluster:
            for t in graph.entity_types[f]:
                if any(t.code == u.code for u in entity_types[rep]):
                    continue
                entity_types[rep].append(t)
            if entity_descriptions is not None:
                assert graph.entity_descriptions is not None
                if rep in entity_descriptions:
                    entity_descriptions[rep] += f" {graph.entity_descriptions[f]}"
                else:
                    entity_descriptions[rep] = graph.entity_descriptions[f]
            for text_id in graph.entity_text_ids[f]:
                if text_id in entity_text_ids[rep]:
                    continue
                entity_text_ids[rep].append(text_id)

    # Prune unknown entity types if they have been resolved after merging
    entity_types = prune_unknown_types(entity_types)

    # Merge the subjects/objects in the triples
    # and update the map to the text ids
    triples: set[Triple] = set()
    triple_text_ids: dict[Triple, list[int]] = {}
    qualifiers: dict[Triple, list[Qualifier]] = defaultdict(list)
    inverted_aliases: dict[str, str] = {}
    for rep, cluster in aliases.items():
        for f in cluster:
            assert f not in inverted_aliases
            inverted_aliases[f] = rep
    assert set(inverted_aliases.keys()) == graph.entities
    for t in graph.triples:
        s, p_id, o = t
        s = inverted_aliases[s]
        o = inverted_aliases[o]
        new_t = (s, p_id, o)
        triples.add(new_t)

        # Adjust the qualifiers
        if graph.qualifiers is not None and t in graph.qualifiers:
            for q in graph.qualifiers[t]:
                q_id, q_o = q
                q_o = inverted_aliases.get(q_o, q_o)
                q = (q_id, q_o)
                if q in qualifiers[new_t]:
                    continue
                qualifiers[new_t].append(q)

        # Adjust the text ids
        for text_id in graph.triple_text_ids[t]:
            if new_t not in triple_text_ids:
                triple_text_ids[new_t] = [text_id]
                continue
            text_ids = triple_text_ids[new_t]
            if text_id not in text_ids:
                text_ids.append(text_id)

    # Construct the new knowledge graph where entities have been deduplicated
    if graph.entity_aliases is not None:
        # Keep the aliases of the input knowledge graph as well, if any
        aliases = {
            e: list(set(g for f in als for g in graph.entity_aliases[f]))
            for e, als in aliases.items()
        }
    graph = Graph(
        entities,
        entity_types,
        entity_descriptions,
        graph.predicates,
        triples,
        entity_text_ids=entity_text_ids,
        triple_text_ids=triple_text_ids,
        qualifiers=qualifiers,
        entity_aliases=aliases,
    )
    return graph
