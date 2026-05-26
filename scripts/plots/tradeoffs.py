import argparse
import json
import os
from typing import Any

import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import pandas as pd

from scripts.plots.plot_utils import get_palette_colors, setup_tueplots

parser = argparse.ArgumentParser(
    prog="Plotter for the 'first page figure'",
)
parser.add_argument("datasets", type=str, help="Datasets separated by a semicolon")
parser.add_argument("llms", type=str, help="LLMs separated by a semicolon")
parser.add_argument("aliases", type=str, help="Aliases separated by a semicolon")
parser.add_argument(
    "--alias-labels",
    type=str,
    help="Labels to use for each alias, separated by a semicolon",
    default="",
)
parser.add_argument(
    "--dataset-labels",
    type=str,
    help="Labels to use for each dataset, separated by a semicolon",
    default="",
)
parser.add_argument(
    "--json-path", type=str, help="The root path to JSON results files", required=True
)

PROMPT_TO_COMPLETION_COST = 0.25

if __name__ == "__main__":
    args = parser.parse_args()
    datasets = args.datasets.split(";")
    llms = args.llms.split(";")
    aliases = args.aliases.split(";")
    if args.alias_labels:
        alias_labels = args.alias_labels.split(";")
    else:
        alias_labels = aliases
    if args.alias_labels:
        dataset_labels = args.dataset_labels.split(";")
    else:
        dataset_labels = datasets
    alias_map = dict(zip(aliases, alias_labels))
    dataset_map = dict(zip(datasets, dataset_labels))

    # Load the relevant metrics from the JSONs
    metrics: list[dict[str, Any]] = []
    for dataset in datasets:
        for llm in llms:
            # Load the patterns hscores
            filepath = os.path.join(
                args.json_path, dataset, llm, "patterns-hscores.json"
            )
            with open(filepath, "r") as fp:
                hscores = json.load(fp)
            for alias in aliases:
                method = alias_map[alias]
                alias_ms = {
                    "dataset": dataset,
                    "llm": llm,
                    "alias": alias,
                    "method": method,
                }
                # Load the computational costs (# facts per 1M prompt (or completion) tokens)
                filepath = os.path.join(
                    args.json_path, dataset, llm, f"kgc-{alias}.json"
                )
                with open(filepath, "r") as fp:
                    results = json.load(fp)
                    num_triples = results["num_triples"]["mean"]
                    num_qualifiers = results["num_qualifiers"]["mean"]
                    num_valid_triples = results["num_valid_triples"]["mean"]
                    num_valid_qualifiers = results["num_valid_qualifiers"]["mean"]
                    ptoks = results["usage"]["cumulative"]["total"]["prompt"]["mean"]
                    ctoks = results["usage"]["cumulative"]["total"]["completion"][
                        "mean"
                    ]
                    ptoks_per_fact = results["usage"]["normalized_facts"]["total"][
                        "prompt"
                    ]["mean"]
                    ctoks_per_fact = results["usage"]["normalized_facts"]["total"][
                        "completion"
                    ]["mean"]
                    alias_ms["wtoks"] = ptoks * PROMPT_TO_COMPLETION_COST + ctoks
                    alias_ms["wtoks_per_fact"] = (
                        ptoks_per_fact * PROMPT_TO_COMPLETION_COST + ctoks_per_fact
                    )
                    alias_ms["ratio_valid_facts"] = (
                        num_valid_triples + num_valid_qualifiers
                    ) / (num_triples + num_qualifiers)
                # Load the QA metrics
                filepath = os.path.join(
                    args.json_path, dataset, llm, f"qa-{alias}.json"
                )
                with open(filepath, "r") as fp:
                    results = json.load(fp)
                    alias_ms["qa_em_mean"] = results["qa"]["exact_match"]["mean"]
                    alias_ms["qa_em_stddev"] = results["qa"]["exact_match"]["stddev"]
                    alias_ms["qa_f1_mean"] = results["qa"]["f1"]["mean"]
                    alias_ms["qa_f1_stddev"] = results["qa"]["f1"]["stddev"]
                # Load the patterns metrics
                if method in hscores["all"]:
                    alias_ms["h_index_mean"] = hscores["all"][method]["h_index"]["mean"]
                    alias_ms["i10_index_mean"] = hscores["all"][method]["i10_index"][
                        "mean"
                    ]
                    alias_ms["i100_index_mean"] = hscores["all"][method]["i100_index"][
                        "mean"
                    ]
                    alias_ms["h_index_std"] = hscores["all"][method]["h_index"]["std"]
                    alias_ms["i10_index_std"] = hscores["all"][method]["i10_index"][
                        "std"
                    ]
                    alias_ms["i100_index_std"] = hscores["all"][method]["i100_index"][
                        "std"
                    ]
                elif method == "KGGen":
                    alias_ms["h_index_mean"] = alias_ms["i10_index_mean"] = alias_ms[
                        "i100_index_mean"
                    ] = 0.0
                    alias_ms["h_index_std"] = alias_ms["i10_index_std"] = alias_ms[
                        "i100_index_std"
                    ] = 0.0
                else:
                    alias_ms["h_index_mean"] = alias_ms["i10_index_mean"] = alias_ms[
                        "i100_index_mean"
                    ] = None
                    alias_ms["h_index_std"] = alias_ms["i10_index_std"] = alias_ms[
                        "i100_index_std"
                    ] = None
                metrics.append(alias_ms)

    setup_tueplots(2, 2, rel_width=1.0, hw_ratio=0.85)
    ax1 = plt.subplot(221)
    ax2 = plt.subplot(222, sharey=ax1)
    ax3 = plt.subplot(223, sharex=ax1)
    ax4 = plt.subplot(224, sharex=ax2, sharey=ax3)
    axs = [[ax1, ax2], [ax3, ax4]]
    plt.subplots_adjust(wspace=0.1, hspace=0.1, top=0.75)

    df = pd.DataFrame(metrics)
    colors = get_palette_colors()
    palette = {alias: c for alias, c in zip(aliases, colors)}
    dataset_markers = ["o", "D"]
    markers = {k: m for k, m in zip(datasets, dataset_markers)}
    for i, llm in enumerate(llms):
        df_llm = df[df["llm"] == llm]
        for (dataset, alias), row in df_llm.groupby(["dataset", "alias"]):
            axs[0][i].errorbar(
                x=row["wtoks_per_fact"],
                # x=row["wtoks"],
                y=row["qa_f1_mean"],
                # yerr=row["qa_f1_stddev"],
                fmt=markers[dataset],
                color=palette[alias],
                # capsize=6,
                markersize=8,
                alpha=0.7,
                label=(dataset, alias),
            )
            axs[1][i].errorbar(
                x=row["wtoks_per_fact"],
                # x=row["wtoks"],
                # y=row["h_index_mean"],
                y=100.0 * row["ratio_valid_facts"],
                # yerr=row["h_index_std"],
                fmt=markers[dataset],
                color=palette[alias],
                # capsize=6,
                markersize=8,
                alpha=0.7,
                label=(dataset, alias),
            )

        axs[0][i].margins(0.15, 0.15)
        axs[0][i].margins(0.15, 0.15)
        axs[1][i].margins(0.15, 0.15)
        axs[1][i].margins(0.15, 0.15)
        axs[0][i].tick_params("x", labelbottom=False)
        if i > 0:
            axs[0][i].tick_params("y", labelleft=False)
            axs[1][i].tick_params("y", labelleft=False)
        axs[1][i].set_xlabel("Tokens cost per fact")

        axs[0][i].grid(linestyle="--", which="major", alpha=0.4, linewidth=0.5)
        axs[1][i].grid(linestyle="--", which="major", alpha=0.4, linewidth=0.5)

    dataset_handles = [
        mlines.Line2D(
            [],
            [],
            color="gray",
            marker=marker,
            linestyle="None",
            label=dataset_map[dataset],
        )
        for dataset, marker in zip(datasets, dataset_markers)
    ]
    method_handles = [
        mpatches.Patch(facecolor=palette[alias], label=alias_map[alias])
        for alias in aliases
    ]
    axs[0][0].legend(
        handles=dataset_handles,
        title="Dataset",
        bbox_to_anchor=(0.0, 1.25),
        loc="lower left",
        borderaxespad=0,
        handletextpad=0.1,
    )
    axs[0][1].legend(
        handles=method_handles,
        title="Method",
        bbox_to_anchor=(-0.5, 1.25),
        loc="lower left",
        borderaxespad=0,
        ncols=2,
        columnspacing=0.3,
        handletextpad=0.5,
    )

    axs[0][0].set_ylabel("QA F1 score")
    # axs[1][0].set_ylabel("H-index BGPs")
    axs[1][0].set_ylabel("% valid facts")
    axs[0][0].set_title("Qwen3-30B-A3B-Instruct")
    axs[0][1].set_title("GPT-OSS-120B")

    plt.subplots_adjust(left=0.125, right=0.97, top=0.75, bottom=0.1025)
    plt.savefig("results/tradeoffs.pdf")
    plt.show()
