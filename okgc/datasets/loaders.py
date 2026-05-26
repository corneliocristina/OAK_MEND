from okgc.datasets.hotpot import load_hotpot1000
from okgc.datasets.musique import load_musique1000
from okgc.datasets.utils import DataEntry
from okgc.utils.sparql import Predicate, TypesHierarchy
from scripts.wdcrawl import (
    load_wd_predicates,
    load_wd_qualifiers,
    load_wd_superclassess,
    load_wd_types,
)

DATASET_NAMES = [
    "hotpot1000",
    "musique1000",
    "fanoutqa-shortest4",
]
DATASET_WITH_GT_NAMES = ["docred", "redocred"]
DATASET_LONG_TEXTS_NAMES = ["fanoutqa-shortest4"]


def load_dataset(name: str, *, datapath: str = "data") -> list[DataEntry]:
    if name == "hotpot1000":
        data = load_hotpot1000(datapath)
    elif name == "musique1000":
        data = load_musique1000(datapath)
    else:
        raise ValueError(f"Unknown dataset '{name}'")
    return data


def load_predicates(
    name: str, *, datapath: str = "data"
) -> tuple[dict[str, Predicate], dict[str, Predicate]]:
    if name in ["hotpot1000", "musique1000"]:
        predicates, inverted_predicates = load_wd_predicates(
            datapath, return_inverted=True
        )
    else:
        raise ValueError(f"Unknown dataset '{name}'")
    return predicates, inverted_predicates


def load_qualifiers(
    name: str, *, datapath: str = "data"
) -> tuple[dict[str, Predicate], dict[str, Predicate]]:
    if name in ["hotpot1000", "musique1000"]:
        qualifiers, inverted_qualifiers = load_wd_qualifiers(
            datapath, return_inverted=True
        )
    else:
        raise ValueError(f"Unknown dataset '{name}'")
    return qualifiers, inverted_qualifiers


def load_types_hierarchy(name: str, *, datapath: str = "data") -> TypesHierarchy:
    assert name in ["hotpot1000", "musique1000"]
    types = load_wd_types(datapath=datapath)
    superclasses = load_wd_superclassess(datapath=datapath)
    types_hierarchy = TypesHierarchy(types, superclasses)
    return types_hierarchy
