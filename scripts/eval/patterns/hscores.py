import argparse
import json
import os
import sys
from collections import defaultdict
from typing import Any

import numpy as np
import pandas as pd

from okgc.datasets.loaders import DATASET_NAMES
from okgc.utils.filesystem import (
    build_results_path,
    format_llm_name,
    load_results_repetitions,
)

parser = argparse.ArgumentParser(
    prog="Query pattern metrics printer",
)
parser.add_argument("dataset", choices=DATASET_NAMES)
parser.add_argument(
    "--aliases",
    type=str,
    required=True,
    help="The aliases of the experiments, separated by semicolon",
)
parser.add_argument(
    "--labels",
    type=str,
    required=True,
    help="The labels for the aliases to put in the bar plot legend, separated by semicolon",
)
parser.add_argument(
    "--llm",
    help="The LLM to use, with the provider",
    default="Qwen/Qwen3-30B-A3B-Instruct-2507-FP8",
)
parser.add_argument(
    "--think",
    help="Whether enable thinking/reasoning has been enabled to generate the KG",
    default=False,
    action="store_true",
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


METHOD_IDS = {"Wikontic": 0, "In-Context": 1, "OAK": 2, "OAK+OC": 3}

ARTIFICIAL_PATTERNS_NAMES = [
    "2p",
    "2i",
    "1p2i",
    "2i1p",
    "3i",
    "3p",
    "r-2i",
    "r-1p2i",
    "r-2i1p",
    "r-3i",
]


def h_index(xs: list[int]) -> int:
    xs = sorted(xs, reverse=True)
    h = 0
    for i, n in enumerate(xs):
        if n < i + 1:
            break
        h = i + 1
    return h


def ik_index(xs: list[int], *, k: int) -> int:
    return np.sum(np.array(xs) >= k).item()


def load_metrics_repetitions_all() -> list[dict[str, Any]]:
    results = []
    for entry in entries:
        # Load the experiments results
        llm_name = format_llm_name(args.llm)
        inner_filepath = build_results_path(
            think=args.think,
        )
        root_path = os.path.join(
            "results",
            "qcounts" if not entry["alias"] else f"qcounts-{entry['alias']}",
            args.dataset,
            llm_name,
            inner_filepath,
        )
        metrics = load_results_repetitions(root_path, verbose=True)
        if not metrics:
            raise RuntimeError("No suitable metrics were found")
        for seed, mss in metrics.items():
            ms = {}
            for xs in mss:
                ms.update(xs)
            for name, res in ms.items():
                r = {
                    "method": entry["id"],
                    "seed": seed,
                    "pattern": name,
                    "counts": np.array(res["counts"]),
                }
                results.append(r)
    return results


def df_multiindex_to_nested(df):
    result = {}
    for row in df.index:
        result[row] = {}
        for top_col in df.columns.get_level_values(0).unique():
            result[row][top_col] = df.loc[row, top_col].to_dict()
    return result


def compute_aggregate_metrics(
    metrics: list[dict[str, Any]], breakdown: bool = False
) -> pd.DataFrame:
    if not breakdown:
        metrics_counts = defaultdict(list)
        for r in metrics:
            metrics_counts[(r["method"], r["seed"])].extend(r["counts"])
        metrics = []
        for (method, seed), counts in metrics_counts.items():
            s = {"method": method, "seed": seed, "counts": counts, "total": sum(counts)}
            metrics.append(s)
    df = pd.DataFrame(metrics)
    df["h_index"] = df.apply(lambda row: h_index(row["counts"]), axis=1)
    for k in [10, 100]:
        df[f"i{k}_index"] = df.apply(lambda row: ik_index(row["counts"], k=k), axis=1)
    if breakdown:
        df = df.groupby(by=["method", "pattern"]).agg(func=metrics_agg_func)
        df = df.sort_values(
            by=["method", "pattern"], key=lambda col: col.map(METHOD_IDS)
        )
    else:
        df = df.groupby(by=["method"]).agg(func=metrics_agg_func)
        df = df.sort_values(by="method", key=lambda col: col.map(METHOD_IDS))
    return df


if __name__ == "__main__":
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 800)
    pd.set_option("display.precision", 1)

    args = parser.parse_args()

    assert args.aliases and args.labels
    aliases = args.aliases.split(";")
    labels = args.labels.split(";")
    assert len(aliases) == len(labels)
    entries = [{"id": l, "alias": a} for a, l in zip(aliases, labels)]

    results = load_metrics_repetitions_all()

    metrics = []
    for r in results:
        s = {
            "method": r["method"],
            "seed": r["seed"],
            "pattern": r["pattern"],
            "counts": r["counts"],
            "total": sum(r["counts"]),
        }
        metrics.append(s)

    metrics_agg_func = {
        k: ["mean", "std"] for k in ["total", "h_index", "i10_index", "i100_index"]
    }

    # Redirect stdout to md file
    if args.md_path:
        md_filepath = os.path.join(args.md_path, "patterns-hscores.md")
        sys.stdout = open(md_filepath, "w")

    # Compute metrics by:
    # (all) combining counts of all patterns
    # (lsq) considering the counts of only the patterns extracted from LSQ-2.0
    # (art_df) considering the counts of only the artificial patterns
    all_df = compute_aggregate_metrics(metrics)
    lsq_df = compute_aggregate_metrics([r for r in metrics if r["pattern"] == "lsq"])
    art_df = compute_aggregate_metrics([r for r in metrics if r["pattern"] != "lsq"])
    brk_df = compute_aggregate_metrics(metrics, breakdown=True)
    print("# All patterns")
    print(all_df)
    print()
    print("# LSQ-2.0 patterns")
    print(lsq_df)
    print()
    print("# Artificial patterns")
    print(art_df)
    print()
    print("# Breadown per pattern structure")
    print(brk_df)
    print()

    def fmt(mean, std):
        if pd.isna(std):
            return f"{mean:.1f} \\pm 0.0"
        return f"{mean:.1f} \\pm {std:.1f}"

    print("# Latex tables")

    print("## All patterns")
    for method, row in all_df.iterrows():
        cells = [
            fmt(row[(m, "mean")], row[(m, "std")])
            for m in ["h_index", "i10_index", "i100_index"]
        ]
        print(f"{method}")
        print("    & " + " & ".join(cells) + " \\\\")

    print()
    print("## LSQ-2.0 patterns")
    for method, row in lsq_df.iterrows():
        cells = [
            fmt(row[(m, "mean")], row[(m, "std")])
            for m in ["h_index", "i10_index", "i100_index"]
        ]
        print(f"{method}")
        print("    & " + " & ".join(cells) + " \\\\")

    print()
    print("## Artificial patterns")
    for method, row in art_df.iterrows():
        cells = [
            fmt(row[(m, "mean")], row[(m, "std")])
            for m in ["h_index", "i10_index", "i100_index"]
        ]
        print(f"{method}")
        print("    & " + " & ".join(cells) + " \\\\")

    # Close the stdout stream and serialize results in JSON
    sys.stdout.close()
    if not args.json_path:
        exit()
    json_filepath = os.path.join(args.json_path, f"patterns-hscores.json")
    with open(json_filepath, "w") as f:
        json_data = {
            "all": df_multiindex_to_nested(all_df),
            "lsq": df_multiindex_to_nested(lsq_df),
            "art": df_multiindex_to_nested(art_df),
        }
        json.dump(json_data, f, indent=2)
