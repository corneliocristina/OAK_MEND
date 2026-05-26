import argparse
import json
import os
import sys
from typing import Any, Callable

import numpy as np
from tqdm import tqdm

from okgc.datasets.loaders import DATASET_NAMES
from okgc.utils.filesystem import (
    build_results_path,
    format_llm_name,
    load_results_repetitions,
)

parser = argparse.ArgumentParser(
    prog="QA results processor", description="Process the results"
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
    "--filter-triples",
    help="Whether to filter out the triples that violate the ontology",
    action="store_true",
    default=False,
)
parser.add_argument(
    "--filter-qualifiers",
    help="Whether to filter out the qualifiers that violate the ontology",
    action="store_true",
    default=False,
)
parser.add_argument(
    "--alias", help="The alias of the input results directory", default="", type=str
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


def compute_metrics(
    rs: list[dict[str, Any]], *, entry_indices: set[int] | None = None
) -> dict[str, Any]:
    # Merge the knowledge graphs in a repetition
    qa_metrics: dict[str, float] = {
        "exact_match": 0.0,
        "precision": 0.0,
        "recall": 0.0,
        "f1": 0.0,
    }
    for result in rs:
        if entry_indices is not None and result["args"]["index"] not in entry_indices:
            continue
        ms = result["qa"]["metrics"]
        for k in ms:
            qa_metrics[k] += ms[k]
    num_results = len(rs)

    metrics = {
        "qa": {
            "exact_match": qa_metrics["exact_match"] / num_results,
            "precision": qa_metrics["precision"] / num_results,
            "recall": qa_metrics["recall"] / num_results,
            "f1": qa_metrics["f1"] / num_results,
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
    results["qa"] = {
        "exact_match": compute_mean_stddev("qa", "exact_match"),
        "precision": compute_mean_stddev("qa", "precision"),
        "recall": compute_mean_stddev("qa", "recall"),
        "f1": compute_mean_stddev("qa", "f1"),
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

    # Load the experiments results
    llm_name = format_llm_name(args.llm)
    inner_filepath = build_results_path(think=args.think)
    alias = "qa" if not args.alias else f"qa-{args.alias}"
    root_path = os.path.join(
        "results",
        alias,
        args.dataset,
        llm_name,
        inner_filepath,
    )
    results = load_results_repetitions(
        root_path,
        f"filter-triples{args.filter_triples}-qualifiers{args.filter_qualifiers}",
    )
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

    # Process each repetition
    metrics: dict[int, dict[str, Any]] = {}
    tqdm_disable = True if num_repetitions == 1 else False
    for seed, rs in tqdm(results.items(), desc="Repetitions", disable=tqdm_disable):
        metrics[seed] = compute_metrics(rs, entry_indices=entry_indices)

    # Average metrics and compute standard deviations
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

    print("# QA Metrics\n")
    print_aggregate_metric("Exact Match", metrics, "qa", "exact_match", precision=3)
    print_aggregate_metric("Precision", metrics, "qa", "precision", precision=3)
    print_aggregate_metric("Recall", metrics, "qa", "recall", precision=3)
    print_aggregate_metric("F1", metrics, "qa", "f1", precision=3)

    # Close the stdout stream and serialize results in JSON
    sys.stdout.close()
    if not args.json_path:
        exit()
    json_filepath = os.path.join(args.json_path, f"{alias}.json")
    with open(json_filepath, "w") as f:
        json.dump(metrics, f, indent=2)
