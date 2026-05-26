import argparse
import json
import os
import sys
from collections import defaultdict
from dataclasses import asdict
from typing import Any, Callable

import numpy as np
import spacy
from dacite import from_dict
from tqdm import tqdm

from okgc.datasets.loaders import (
    DATASET_NAMES,
    load_dataset,
    load_predicates,
    load_qualifiers,
    load_types_hierarchy,
)
from okgc.utils.filesystem import (
    build_results_path,
    format_llm_name,
    load_results_repetitions,
)
from okgc.utils.graph import Graph, Triple, dict_to_graph
from okgc.utils.sparql import Predicate
from okgc.utils.transform import TransformCounters
from okgc.utils.typecheck import (
    satisfy_domain_range_constraint,
    satisfy_qualifier_constraints,
)
from okgc.utils.usage import UsageInfo

parser = argparse.ArgumentParser(
    prog="KGC results processor", description="Process the results"
)
parser.add_argument("dataset", choices=DATASET_NAMES)
parser.add_argument(
    "--llm",
    help="The LLM being used during KG construction",
    default="Qwen/Qwen3-30B-A3B-Instruct-2507-FP8",
)
parser.add_argument(
    "--think",
    help="Whether thinking/reasoning has been enabled during KG construction",
    default=False,
    action="store_true",
)
parser.add_argument(
    "--in-alias", help="The alias of the input results directory", default="", type=str
)
parser.add_argument(
    "--md-path",
    help="The directory containing Markdown files of the results, if any",
    default="",
    type=str,
)
parser.add_argument(
    "--json-path",
    help="The directory containing JSON files of the results, if any",
    default="",
    type=str,
)


def compute_ontology_consistency_metrics(kg: Graph) -> dict[str, Any]:
    # Compute the number of valid triples and the mistakes
    num_valid_triples = 0
    num_valid_qualifiers = 0
    ls_invalid_subject: list[dict[str, Any]] = []
    ls_invalid_object: list[dict[str, Any]] = []
    ls_invalid_qualifier: list[dict[str, Any]] = []
    for t in kg.triples:
        s, p_id, o = t
        p = kg.predicates[p_id]
        valid_subject_type, valid_object_type = satisfy_domain_range_constraint(
            p,
            subject_types=kg.entity_types[s],
            object_types=kg.entity_types[o],
            hierarchy=types_hierarchy,
        )
        if not valid_subject_type:
            entry = {
                "text_ids": kg.triple_text_ids[t],
                "triple": t,
                "expected_types": [types_hierarchy[c].name for c in p.domain],
                "found_types": [t.name for t in kg.entity_types[s]],
            }
            ls_invalid_subject.append(entry)
        if not valid_object_type:
            entry = {
                "text_ids": kg.triple_text_ids[t],
                "triple": t,
                "expected_types": [types_hierarchy[c].name for c in p.range],
                "found_types": [t.name for t in kg.entity_types[o]],
            }
            ls_invalid_object.append(entry)
        if valid_subject_type and valid_object_type:
            num_valid_triples += 1
        if kg.qualifiers is not None and t in kg.qualifiers:
            for q in kg.qualifiers[t]:
                q_id, q_o = q
                q_p = kg.predicates[q_id]
                if (
                    q_o not in kg.entity_types
                ):  # E.g., in Wikontic no types are assigned to qualifier's object
                    continue  # So, we skip the check
                valid_predicate, valid_object_type = satisfy_qualifier_constraints(
                    p,
                    q_p,
                    kg.entity_types[q_o],
                    hierarchy=types_hierarchy,
                )
                if valid_predicate and valid_object_type:
                    num_valid_qualifiers += 1
                    continue
                q_id, q_o = q
                expected_qualifiers: list[str] | None = None
                if p.qualifiers is not None:
                    expected_qualifiers = [
                        qualifiers[w_id].name for w_id in p.qualifiers
                    ]
                entry = {
                    "text_ids": kg.triple_text_ids[t],
                    "triple": t,
                    "expected_qualifiers": expected_qualifiers,
                    "expected_object_types": [
                        types_hierarchy[c].name for c in q_p.range
                    ],
                    "object": q_o,
                    "found_object_types": [t.name for t in kg.entity_types[q_o]],
                    "found_qualifier_id": q_p.code,
                    "found_qualifier": q_p.name,
                }
                ls_invalid_qualifier.append(entry)
    return {
        "num_valid_triples": num_valid_triples,
        "num_valid_qualifiers": num_valid_qualifiers,
        "ls_invalid_subject": ls_invalid_subject,
        "ls_invalid_object": ls_invalid_object,
        "ls_invalid_qualifier": ls_invalid_qualifier,
    }


def compute_metrics(
    rs: list[dict[str, Any]], *, entry_indices: set[int] | None = None
) -> dict[str, Any]:
    num_triples = 0
    num_qualifiers = 0
    # The usage information
    generation_usage_info: dict[str, UsageInfo] = defaultdict(UsageInfo)
    corrections_usage_info = UsageInfo()
    # The transform counters
    triple_transform_counters = TransformCounters()
    qualifier_transform_counters = TransformCounters()
    # Ontology consistency metrics accumulators
    num_valid_triples = 0
    num_valid_qualifiers = 0
    ls_invalid_subject: list[dict[str, Any]] = []
    ls_invalid_object: list[dict[str, Any]] = []
    ls_invalid_qualifier: list[dict[str, Any]] = []
    # Structural metrics
    unique_entities: set[str] = set()
    unique_predicates: dict[str, Predicate] = dict()
    in_out_relationships: dict[str, set[Triple]] = defaultdict(set)
    entities_per_predicate: dict[str, set[str]] = defaultdict(set)
    entity_pairs_predicates: dict[tuple[str, str], set[str]] = defaultdict(set)
    predicate_combinations: dict[tuple[str, ...], set[tuple[str, str]]] = defaultdict(
        set
    )
    # Retrieve or compute the metrics
    tqdm_position = 1 if num_repetitions > 1 else 0
    tqdm_leave = True if num_repetitions == 1 else False
    tqdm_results = tqdm(rs, desc="KGs", position=tqdm_position, leave=tqdm_leave)
    for result in tqdm_results:
        if entry_indices is not None and result["args"]["index"] not in entry_indices:
            continue
        kg = dict_to_graph(result["graph"])
        # Get usage information and corrections information
        for k, m in result["usage"]["generation"].items():
            generation_usage_info[k] += from_dict(UsageInfo, data=m)
        if "corrections" in result:
            corrections = result["corrections"]
            corrections_usage_info += from_dict(
                UsageInfo, data=result["usage"]["corrections"]
            )
            triple_transform_counters += from_dict(
                TransformCounters,
                data=corrections["transform_counters"]["triples"],
            )
            qualifier_transform_counters += from_dict(
                TransformCounters,
                data=corrections["transform_counters"]["qualifiers"],
            )
        # Accumulate the number of triples and qualifiers
        num_triples += len(kg.triples)
        if kg.qualifiers is not None:
            num_qualifiers += sum(
                len(kg.qualifiers[t]) for t in kg.triples if t in kg.qualifiers
            )
        # Evaluate the ontology consistency
        consistency_matrics = compute_ontology_consistency_metrics(kg)
        num_valid_triples += consistency_matrics["num_valid_triples"]
        num_valid_qualifiers += consistency_matrics["num_valid_qualifiers"]
        ls_invalid_subject.extend(consistency_matrics["ls_invalid_subject"])
        ls_invalid_object.extend(consistency_matrics["ls_invalid_object"])
        ls_invalid_qualifier.extend(consistency_matrics["ls_invalid_qualifier"])
        # Update the structural metrics
        unique_entities |= kg.entities
        unique_predicates.update(kg.predicates)
        for t in kg.triples:
            s, r, o = t
            # p = kg.predicates[r]
            # ok_subject, ok_object = satisfy_domain_range_constraint(
            #     p,
            #     kg.entity_types[s],
            #     kg.entity_types[o],
            #     hierarchy=types_hierarchy,
            # )
            # if not ok_subject or not ok_object:
            #     continue
            in_out_relationships[s].add(t)
            in_out_relationships[o].add(t)
            entities_per_predicate[r].add(s)
            entities_per_predicate[r].add(o)
            entity_pairs_predicates[(s, o)].add(r)
    for (s, o), rs in entity_pairs_predicates.items():
        filtered_ps = [
            repr(unique_predicates[r])
            for r in rs
            if not unique_predicates[r].is_unknown()
        ]
        if not filtered_ps:
            continue
        filtered_ps = tuple(sorted(filtered_ps))
        predicate_combinations[filtered_ps].add((s, o))

    # Compute some structural metrics
    if len(entity_pairs_predicates) > 0:
        avg_entity_degree = sum(map(len, in_out_relationships.values())) / len(
            in_out_relationships
        )
        avg_predicate_degree = sum(map(len, entities_per_predicate.values())) / len(
            entities_per_predicate
        )
        avg_edge_multiplicity = sum(map(len, entity_pairs_predicates.values())) / len(
            entity_pairs_predicates
        )
        max_edge_multiplicity = max(map(len, entity_pairs_predicates.values()))
        simplification_index = 1.0 - 1.0 / avg_edge_multiplicity
        parallel_link_fraction = sum(
            1 if len(ps) > 1 else 0 for ps in entity_pairs_predicates.values()
        ) / len(entity_pairs_predicates)
    else:
        avg_entity_degree = 0.0
        avg_predicate_degree = 0.0
        avg_edge_multiplicity = 0.0
        max_edge_multiplicity = 0
        simplification_index = 0.0
        parallel_link_fraction = 0.0

    # Compute additional metrics
    num_facts = num_triples + num_qualifiers
    num_valid_facts = num_valid_triples + num_valid_qualifiers
    total_usage_info = (
        sum(generation_usage_info.values(), start=UsageInfo()) + corrections_usage_info
    )

    metrics = {
        "num_triples": num_triples,
        "num_valid_triples": num_valid_triples,
        "num_invalid_triples": num_triples - num_valid_triples,
        "ratio_valid_triples": 0.0
        if num_triples == 0
        else num_valid_triples / num_triples,
        "num_qualifiers": num_qualifiers,
        "num_valid_qualifiers": num_valid_qualifiers,
        "num_invalid_qualifiers": num_qualifiers - num_valid_qualifiers,
        "ratio_valid_qualifiers": 0.0
        if num_qualifiers == 0
        else num_valid_qualifiers / num_qualifiers,
        "mistakes": {
            "invalid_subject": ls_invalid_subject,
            "invalid_object": ls_invalid_object,
            "invalid_qualifier": ls_invalid_qualifier,
        },
        "transform_counters": {
            "triples": asdict(triple_transform_counters),
            "qualifiers": asdict(qualifier_transform_counters),
        },
        "structure_metrics": {
            "num_entities": len(unique_entities),
            "num_predicates": len(unique_predicates),
            "avg_entity_degree": avg_entity_degree,
            "avg_predicate_degree": avg_predicate_degree,
            "multiplicity_metrics": {
                "mean": avg_edge_multiplicity,
                "max": max_edge_multiplicity,
                "simplification_index": simplification_index,
                "parallel_link_fraction": parallel_link_fraction,
                "predicate_combinations": predicate_combinations,
            },
        },
        "usage": {
            "generation_usage_info_keys": list(generation_usage_info.keys()),
        },
    }

    for name, div in zip(
        [
            "cumulative",
            "normalized_triples",
            "normalized_facts",
            "normalized_valid_triples",
            "normalized_valid_facts",
        ],
        [1, num_triples, num_facts, num_valid_triples, num_valid_facts],
    ):
        metrics["usage"][name] = {
            "generation": {
                k: {
                    "prompt": 0 if div == 0 else usage_info.prompt_tokens / div,
                    "completion": 0 if div == 0 else usage_info.completion_tokens / div,
                }
                for k, usage_info in generation_usage_info.items()
            },
            "corrections": {
                "prompt": 0 if div == 0 else corrections_usage_info.prompt_tokens / div,
                "completion": 0
                if div == 0
                else corrections_usage_info.completion_tokens / div,
            },
            "total": {
                "prompt": 0 if div == 0 else total_usage_info.prompt_tokens / div,
                "completion": 0
                if div == 0
                else total_usage_info.completion_tokens / div,
            },
        }

    return metrics


def aggregate_metrics(metrics: dict[int, dict[str, Any]]) -> dict[str, Any]:
    def collect_metrics(*names: str) -> list[Any]:
        assert len(names) > 0
        ls = []
        for _, ds in metrics.items():
            d = ds
            for name in names:
                d = d[name]
            ls.append(d)
        return ls

    def compute_mean_stddev(
        *names: str, map_fn: Callable | None = None
    ) -> dict[str, Any]:
        ms = collect_metrics(*names)
        if map_fn is not None:
            ms = [map_fn(m) for m in ms]
        return {"mean": np.mean(ms), "stddev": np.std(ms) if len(ms) > 1 else 0.0}

    results: dict[str, Any] = {}
    results["num_triples"] = compute_mean_stddev("num_triples")
    results["num_valid_triples"] = compute_mean_stddev("num_valid_triples")
    results["num_invalid_triples"] = compute_mean_stddev("num_invalid_triples")
    results["ratio_valid_triples"] = compute_mean_stddev("ratio_valid_triples")
    results["num_qualifiers"] = compute_mean_stddev("num_qualifiers")
    results["num_valid_qualifiers"] = compute_mean_stddev("num_valid_qualifiers")
    results["num_invalid_qualifiers"] = compute_mean_stddev("num_invalid_qualifiers")
    results["ratio_valid_qualifiers"] = compute_mean_stddev("ratio_valid_qualifiers")
    results["structure_metrics"] = {
        "num_entities": compute_mean_stddev("structure_metrics", "num_entities"),
        "num_predicates": compute_mean_stddev("structure_metrics", "num_predicates"),
        "avg_entity_degree": compute_mean_stddev(
            "structure_metrics", "avg_entity_degree"
        ),
        "avg_predicate_degree": compute_mean_stddev(
            "structure_metrics", "avg_predicate_degree"
        ),
        "multiplicity_metrics": {
            "mean": compute_mean_stddev(
                "structure_metrics", "multiplicity_metrics", "mean"
            ),
            "max": compute_mean_stddev(
                "structure_metrics", "multiplicity_metrics", "max"
            ),
            "simplification_index": compute_mean_stddev(
                "structure_metrics", "multiplicity_metrics", "simplification_index"
            ),
            "parallel_link_fraction": compute_mean_stddev(
                "structure_metrics", "multiplicity_metrics", "parallel_link_fraction"
            ),
            "predicate_combinations": metrics[list(metrics.keys())[0]][
                "structure_metrics"
            ]["multiplicity_metrics"]["predicate_combinations"],
        },
    }
    results["mistakes"] = {
        "num_invalid_subject": compute_mean_stddev(
            "mistakes", "invalid_subject", map_fn=len
        ),
        "num_invalid_object": compute_mean_stddev(
            "mistakes", "invalid_object", map_fn=len
        ),
    }

    results["usage"] = {}
    for name in [
        "cumulative",
        "normalized_triples",
        "normalized_facts",
        "normalized_valid_triples",
        "normalized_valid_facts",
    ]:
        results["usage"][name] = {
            "generation": {
                k: {
                    "prompt": compute_mean_stddev(
                        "usage", name, "generation", k, "prompt"
                    ),
                    "completion": compute_mean_stddev(
                        "usage", name, "generation", k, "completion"
                    ),
                }
                for k in generation_usage_info_keys
            },
            "corrections": {
                "prompt": compute_mean_stddev("usage", name, "corrections", "prompt"),
                "completion": compute_mean_stddev(
                    "usage", name, "corrections", "completion"
                ),
            },
            "total": {
                "prompt": compute_mean_stddev("usage", name, "total", "prompt"),
                "completion": compute_mean_stddev("usage", name, "total", "completion"),
            },
        }

    results["transform_counters"] = {
        "triples": {
            k: compute_mean_stddev("transform_counters", "triples", k)
            for k in [
                "swap_subject_object",
                "add_subject_type",
                "add_object_type",
                "replace_predicate",
                "replace_subject",
                "replace_object",
            ]
        },
        "qualifiers": {
            k: compute_mean_stddev("transform_counters", "qualifiers", k)
            for k in ["add_object_type", "replace_predicate"]
        },
    }
    return results


def print_aggregate_metric(
    description: str, metrics: dict[str, Any], *names: str, precision: int = 0
):
    d = metrics
    for name in names:
        d = d[name]
    assert "mean" in d and "stddev" in d
    mean = d["mean"]
    stddev = d["stddev"]
    if precision > 0:
        print(f"{description}:", f"{mean:.0{precision}f} +- {stddev:.0{precision}f}\n")
        return
    mean = int(np.round(mean))
    stddev = int(np.round(stddev))
    print(f"{description}:", f"{mean} +- {stddev}\n")


if __name__ == "__main__":
    args = parser.parse_args()

    # Load the data, the predicates, and the types
    data = load_dataset(args.dataset)
    predicates, inverted_predicates = load_predicates(args.dataset)
    qualifiers, inverted_qualifiers = load_qualifiers(args.dataset)
    types_hierarchy = load_types_hierarchy(args.dataset)

    # Load the experiments results
    llm_name = format_llm_name(args.llm)
    inner_filepath = build_results_path(
        think=args.think,
    )
    alias = "kgc" if not args.in_alias else f"kgc-{args.in_alias}"
    root_path = os.path.join(
        "results",
        alias,
        args.dataset,
        llm_name,
        inner_filepath,
    )
    results = load_results_repetitions(root_path)
    if not results:
        raise RuntimeError("No suitable results were found")
    num_repetitions = len(results)
    num_entries = [len(rs) for seed, rs in results.items()]
    all_entry_indices = [
        set(res["args"]["index"] for res in rs) for seed, rs in results.items()
    ]
    entry_indices = all_entry_indices[0]
    for idx in all_entry_indices[1:]:
        entry_indices &= idx
    num_entries_per_repetitions = len(entry_indices)

    # Load nlp pipeline
    nlp = spacy.load("en_core_web_sm")

    # Process each repetition
    metrics: dict[int, dict[str, Any]] = {}
    tqdm_disable = num_repetitions == 1
    for seed, rs in tqdm(
        results.items(),
        desc="Repetitions",
        position=0,
        leave=True,
        disable=tqdm_disable,
    ):
        metrics[seed] = compute_metrics(rs, entry_indices=entry_indices)

    # Average metrics and compute standard deviations
    generation_usage_info_keys = list(metrics.values())[0]["usage"][
        "generation_usage_info_keys"
    ]
    metrics: dict[str, Any] = aggregate_metrics(metrics)

    # Redirect stdout to md file
    if args.md_path:
        md_filepath = os.path.join(args.md_path, f"{alias}.md")
        sys.stdout = open(md_filepath, "w")

    print("# Settings\n")
    print(f"Dataset used: {args.dataset}\n")
    print(f"LLM used: {args.llm}\n")
    print(f"Think: {args.think}\n")
    print(f"Number of repetitions processed: {len(results)}\n")
    print(f"Number of entries per repetition: {num_entries_per_repetitions}\n")

    print("# Statistics\n")
    print_aggregate_metric("Total number of triples generated", metrics, "num_triples")
    print_aggregate_metric(
        "Total number of triples generated (filtered)", metrics, "num_valid_triples"
    )
    print_aggregate_metric(
        "Ratio of valid triples generated", metrics, "ratio_valid_triples", precision=3
    )
    print_aggregate_metric(
        "Total number of inconsistent triples", metrics, "num_invalid_triples"
    )
    print_aggregate_metric(
        "Total number of qualifiers generated", metrics, "num_qualifiers"
    )
    print_aggregate_metric(
        "Total number of qualifiers generated (filtered)",
        metrics,
        "num_valid_qualifiers",
    )
    print_aggregate_metric(
        "Ratio of valid qualifiers generated",
        metrics,
        "ratio_valid_qualifiers",
        precision=3,
    )
    print_aggregate_metric(
        "Total number of inconsistent qualifiers", metrics, "num_invalid_qualifiers"
    )

    print("# Computational Cost\n")

    def print_token_costs_info(*method_step: str) -> None:
        for token_kind in ["prompt", "completion"]:
            print_aggregate_metric(
                f"Total number of {token_kind} tokens",
                metrics,
                "usage",
                f"cumulative",
                *method_step,
                f"{token_kind}",
            )
        for normalization_kind in ["triple", "fact"]:
            for token_kind in ["prompt", "completion"]:
                print_aggregate_metric(
                    f"Number of {token_kind} tokens per {normalization_kind}",
                    metrics,
                    "usage",
                    f"normalized_{normalization_kind}s",
                    *method_step,
                    f"{token_kind}",
                    precision=1,
                )
        for normalization_kind in ["triple", "fact"]:
            for token_kind in ["prompt", "completion"]:
                print_aggregate_metric(
                    f"Number of {token_kind} tokens per valid {normalization_kind}",
                    metrics,
                    "usage",
                    f"normalized_valid_{normalization_kind}s",
                    *method_step,
                    f"{token_kind}",
                    precision=1,
                )

    print("## Breakdown of Generation-only Costs\n")
    for k in generation_usage_info_keys:
        print(f"### Cost Category: {k}\n")
        print_token_costs_info("generation", k)

    print("## Corrections\n")
    print_token_costs_info("corrections")

    print("## Total (Generation + Corrections)\n")
    print_token_costs_info("total")

    print("# Ontology Violation Metrics\n")
    print_aggregate_metric(
        "Number of triples having invalid subject",
        metrics,
        "mistakes",
        "num_invalid_subject",
    )
    print_aggregate_metric(
        "Number of triples having invalid object",
        metrics,
        "mistakes",
        "num_invalid_object",
    )

    print("# Structure and Multiplicity Metrics\n")

    print_aggregate_metric(
        f"Num. of unique entities",
        metrics,
        "structure_metrics",
        "num_entities",
        precision=1,
    )
    print_aggregate_metric(
        f"Num. of unique predicates",
        metrics,
        "structure_metrics",
        "num_predicates",
        precision=1,
    )
    print_aggregate_metric(
        f"Avg. entity degree",
        metrics,
        "structure_metrics",
        "avg_entity_degree",
        precision=2,
    )
    print_aggregate_metric(
        f"Avg. predicate degree",
        metrics,
        "structure_metrics",
        "avg_predicate_degree",
        precision=2,
    )
    print_aggregate_metric(
        f"Avg. number of links between connected pairs of entities (or Avg. edge multiplicity)",
        metrics,
        "structure_metrics",
        "multiplicity_metrics",
        "mean",
        precision=2,
    )
    print_aggregate_metric(
        f"Max. number of links between connected pairs of entities (or edge multiplicity)",
        metrics,
        "structure_metrics",
        "multiplicity_metrics",
        "max",
        precision=2,
    )
    print_aggregate_metric(
        f"How far is the multi-graph from being a graph (range is [0-1))",
        metrics,
        "structure_metrics",
        "multiplicity_metrics",
        "simplification_index",
        precision=3,
    )
    print_aggregate_metric(
        f"Ratio of connected pairs of entities that are linked by 2 or more edges (or parallel link fraction)",
        metrics,
        "structure_metrics",
        "multiplicity_metrics",
        "parallel_link_fraction",
        precision=3,
    )
    predicate_combinations = metrics["structure_metrics"]["multiplicity_metrics"][
        "predicate_combinations"
    ]
    if predicate_combinations:
        sorted_predicate_combinations = sorted(
            predicate_combinations.items(), key=lambda x: len(x[1]), reverse=True
        )
        top_predicate_combinations = list(
            filter(lambda x: len(x[0]) > 1, sorted_predicate_combinations)
        )[:10]
        print(f"Top 10 predicate combinations connecting pairs of entities:")
        for qs, pairs in top_predicate_combinations:
            print(f"- (N = {len(pairs)}) {qs}\n  Entity pairs examples:")
            pairs_list = list(pairs)
            for pair in pairs_list[:5]:
                print(f"    - {pair}")
        print()
    del metrics["structure_metrics"]["multiplicity_metrics"]["predicate_combinations"]

    print("# LLM Corrections Transform Counters\n")
    print("## Triples\n")
    for counter in metrics["transform_counters"]["triples"]:
        counter_label = counter.replace("_", " ").capitalize()
        print_aggregate_metric(
            f"Transformation '{counter_label}' counter",
            metrics,
            "transform_counters",
            "triples",
            counter,
        )
    print("## Qualifiers\n")
    for counter in metrics["transform_counters"]["qualifiers"]:
        counter_label = counter.replace("_", " ").capitalize()
        print_aggregate_metric(
            f"Transformation '{counter_label}' counter",
            metrics,
            "transform_counters",
            "qualifiers",
            counter,
        )

    # Close the stdout stream and serialize results in JSON
    sys.stdout.close()
    if not args.json_path:
        exit()
    json_filepath = os.path.join(args.json_path, f"{alias}.json")
    with open(json_filepath, "w") as f:
        json.dump(metrics, f, indent=2)
