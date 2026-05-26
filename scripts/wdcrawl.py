import json
import os
import re
import traceback
from collections import defaultdict
from dataclasses import asdict

from dacite import Config, from_dict

from okgc.utils.sparql import (
    Predicate,
    PredicateKind,
    TypeInfo,
    collect_superclasses,
    lookup_property_by_code,
    lookup_type_by_code,
)


def _load_predicates(
    filepath: str, *, return_inverted: bool = False
) -> dict[str, Predicate] | tuple[dict[str, Predicate], dict[str, Predicate]]:
    predicates: dict[str, Predicate] = {}
    with open(filepath, "r") as fp:
        data = json.load(fp)
        for code, info in data.items():
            predicates[code] = from_dict(
                Predicate, data=info, config=Config(cast=[PredicateKind])
            )
    num_predicates = len(predicates)
    if not return_inverted:
        return predicates
    inverted_predicates = {p.name: p for _, p in predicates.items()}
    assert len(inverted_predicates) == num_predicates
    return predicates, inverted_predicates


def load_wd_predicates(
    datapath: str = "data", *, return_inverted: bool = False
) -> dict[str, Predicate] | tuple[dict[str, Predicate], dict[str, Predicate]]:
    return _load_predicates(
        os.path.join(datapath, "wikidata", "wdpredicates.json"),
        return_inverted=return_inverted,
    )


def load_wd_qualifiers(
    datapath: str = "data", *, return_inverted: bool = False
) -> dict[str, Predicate] | tuple[dict[str, Predicate], dict[str, Predicate]]:
    return _load_predicates(
        os.path.join(datapath, "wikidata", "wdqualifiers.json"),
        return_inverted=return_inverted,
    )


def load_wd_types(datapath: str = "data") -> dict[str, TypeInfo]:
    with open(os.path.join(datapath, "wikidata", "wdtypes.json"), "r") as fp:
        raw_types = json.load(fp)
    return {c: from_dict(TypeInfo, data=t) for c, t in raw_types.items()}


def load_wd_superclassess(datapath: str = "data") -> dict[str, set[str]]:
    with open(os.path.join(datapath, "wikidata", "wdsuperclasses.json"), "r") as fp:
        superclasses = json.load(fp)
    return {t: set(ss) for t, ss in superclasses.items()}


if __name__ == "__main__":
    from tqdm.contrib.concurrent import thread_map

    def lookup_property(code: str) -> Predicate | None:
        p: Predicate | None
        try:
            p = lookup_property_by_code(code)
        except Exception:
            print(
                f"Lookup for property '{code}' failed. Reason: {traceback.format_exc()}"
            )
            p = None
        return p

    def lookup_type(code: str) -> TypeInfo | None:
        t: TypeInfo | None
        try:
            t = lookup_type_by_code(code)
        except Exception:
            print(f"Lookup for item '{code}' failed. Reason: {traceback.format_exc()}")
            t = None
        return t

    max_workers = 1

    # Find the information about the predicates
    with open(os.path.join("data", "wikidata", "predicates.json"), "r") as fp:
        codes = json.load(fp)
    predicates: dict[str, Predicate] = {}
    properties = thread_map(
        lookup_property, codes, max_workers=max_workers, desc="Predicates"
    )
    for code, p in zip(codes, properties):
        if p is None:
            continue
        predicates[code] = p
    with open(os.path.join("data", "wikidata", "wdpredicates.json"), "w") as fp:
        formatted_predicates: dict[str, dict] = {}
        for code, p in predicates.items():
            formatted_predicates[code] = asdict(p)
        json.dump(formatted_predicates, fp)

    # Find information about the qualifiers, if not already in the predicates dictionary
    qualifiers: dict[str, Predicate] = {}
    missed_qualifiers: set[str] = set()
    for code, p in predicates.items():
        if p.qualifiers is None:
            continue
        for q_code in p.qualifiers:
            if q_code in qualifiers:
                continue
            if q_code in predicates:
                qualifiers[q_code] = predicates[q_code]
                continue
            missed_qualifiers.add(q_code)
    for p in thread_map(
        lookup_property, missed_qualifiers, max_workers=max_workers, desc="Qualifiers"
    ):
        if p is None:
            continue
        if p.code in qualifiers:
            continue
        qualifiers[p.code] = p
    with open(os.path.join("data", "wikidata", "wdqualifiers.json"), "w") as fp:
        formatted_qualifiers: dict[str, dict] = {}
        for code, p in qualifiers.items():
            formatted_qualifiers[code] = asdict(p)
        json.dump(formatted_qualifiers, fp)

    # Find the information about the types
    predicates = load_wd_predicates()
    qualifiers = load_wd_qualifiers()
    type_codes: set[str] = set()
    for p in predicates.values():
        for c in p.domain + p.range:
            type_codes.add(c)
    for p in qualifiers.values():
        for c in p.domain + p.range:
            type_codes.add(c)
    types: dict[str, TypeInfo] = {}
    retrieved_types: list[TypeInfo | None] = thread_map(
        lookup_type, type_codes, max_workers=max_workers, desc="Types"
    )
    type_label_to_codes: dict[str, list[str]] = defaultdict(list)
    for t in retrieved_types:
        if t is None:
            continue
        type_label_to_codes[t.name].append(t.code)
    for t in retrieved_types:
        if t is None:
            continue
        if len(type_label_to_codes[t.name]) > 1:
            assert t.description is not None
            brief_description = re.sub(
                r"\(.+\)", "", t.description, flags=re.DOTALL
            ).strip()
            brief_description = re.sub(" +", " ", brief_description)
            extended_aliases = list(t.aliases) if t.aliases else []
            if t.name not in extended_aliases:
                extended_aliases.append(t.name)
            t = TypeInfo(
                t.code,
                name=f"{t.name} ({brief_description})",
                description=t.description,
                aliases=extended_aliases,
            )
        types[t.code] = t
    formatted_types = {c: asdict(t) for c, t in types.items()}
    with open(os.path.join("data", "wikidata", "wdtypes.json"), "w") as fp:
        json.dump(formatted_types, fp)

    # Collect the type superclasses
    superclasses: dict[str, set[str]] = {}
    types = load_wd_types()
    types_superclasses = thread_map(
        collect_superclasses,
        [types[k] for k in types],
        max_workers=max_workers,
        desc="Type superclasses",
    )
    for c, ss in zip(types.keys(), types_superclasses):
        superclasses[c] = set(s for s in ss if s in types)
    formatted_superclasses = {c: list(ss) for c, ss in superclasses.items()}
    with open(os.path.join("data", "wikidata", "wdsuperclasses.json"), "w") as fp:
        json.dump(formatted_superclasses, fp)

    # Double check all types and all predicates have unique names as well
    assert len(set(t.name for t in types.values())) == len(types)
    predicates = predicates.copy()
    predicates.update(qualifiers)
    assert len(set(p.name for p in predicates.values())) == len(predicates)
