import argparse
import json
import os
import time
import traceback
from collections import defaultdict
from dataclasses import asdict
from typing import Any

import dspy
from sentence_transformers import SentenceTransformer
from tqdm.contrib.concurrent import thread_map

from okgc.datasets.loaders import (
    DATASET_NAMES,
    load_dataset,
)
from okgc.kg_gen import KGGen
from okgc.kg_gen.models import Graph as KGGenGraph
from okgc.utils.filesystem import build_results_path, format_llm_name
from okgc.utils.graph import Graph, graph_to_dict
from okgc.utils.openai_wrapper import OpenAIClient
from okgc.utils.sparql import Predicate, TypeInfo
from okgc.utils.usage import UsageInfo
from scripts.utils import ranges_to_indices

# Configure cache settings
dspy.configure_cache(
    enable_disk_cache=False,
    enable_memory_cache=False,
)

parser = argparse.ArgumentParser(
    prog="Experiment Launcher",
    description="Run an experiment and store the results to disk",
)
parser.add_argument("dataset", choices=DATASET_NAMES)
parser.add_argument(
    "--ranges",
    help="The index of the entry in the selected dataset",
    type=str,
    required=True,
)
parser.add_argument(
    "--llm-api-base",
    help="The LLM API url including the port",
    default="http://localhost:9110/v1",
)
parser.add_argument(
    "--llm",
    help="The LLM to use. If empty, then it defaults to the first model available on the endpoint",
    default="",
)
parser.add_argument(
    "--llm-provider",
    help="The provider of the LLM, to be used with LiteLLM",
    default="hosted_vllm",
)
parser.add_argument(
    "--sent-embed",
    help="The sentence embedding model to use",
    default="all-MiniLM-L6-v2",
)
parser.add_argument(
    "--api-key", help="The API key", default="okgc-reflect-key-deadbeef"
)
parser.add_argument(
    "--out-alias",
    help="The alias of the output results directory",
    default="",
    type=str,
)
parser.add_argument(
    "--seed",
    help="The seed to pass to the LLM",
    default=42,
    type=int,
)
parser.add_argument(
    "--n-jobs", help="The number of jobs for multiprocessing", default=1, type=int
)
parser.add_argument("--verbose", default=False, action="store_true")


def run_experiment(entry_idx: int):
    result_filename = f"entry-{entry_idx:04}.json"
    result_filepath = os.path.join(outputs_path, result_filename)
    if os.path.isfile(result_filepath):
        print(f"Skipping entry index {entry_idx:04} ...")
        return
    if entry_idx not in range(len(data)):
        raise ValueError(f"Entry index out of range {{0, ..., {len(data) - 1}}}")
    entry = data[entry_idx]

    # Instantiate the KGGen object
    kggen = KGGen(
        # Set the models and API bases
        model=f"{args.llm_provider}/{llm_identifier}",
        retrieval_model=sent_embed,
        api_base=args.llm_api_base,
        api_key=args.api_key,
        seed=args.seed,
    )

    # Generate the knowledge graph
    start_time = time.perf_counter()
    kg: KGGenGraph
    usage_info: dict[str, UsageInfo] = defaultdict(UsageInfo)
    assert len(entry.texts) > 0
    try:
        # Generate the KG fragments
        kgs: list[KGGenGraph] = []
        for text in entry.texts:
            # Deduplicate later, after KG aggregation
            kggen.reset_token_usage()
            kg = kggen.generate(text, deduplication_method=None)
            kgs.append(kg)
            token_usage = kggen.extract_token_usage_from_history()
            usage_info["construction"] += UsageInfo(
                token_usage["prompt_tokens"], token_usage["completion_tokens"]
            )
        # Aggregate the KGs
        kg = kggen.aggregate(kgs)
        # Deduplicate entities and predicates
        kggen.reset_token_usage()
        kg = kggen.deduplicate(kg)
        token_usage = kggen.extract_token_usage_from_history()
        usage_info["deduplication"] = UsageInfo(
            token_usage["prompt_tokens"], token_usage["completion_tokens"]
        )
    except Exception:
        print(f"Failed entry index {entry_idx}. Message:\n{traceback.format_exc()}")
        return

    # Convert from the KGGen graph format to our
    # By doing so, we will re-use the mutli-step QA algorithm implemented by Wikontic
    entities = set(s for s, _, _ in kg.relations) | set(o for _, _, o in kg.relations)
    formatted_kg = Graph(
        entities,
        entity_types={e: [TypeInfo.unknown_from_name("Entity")] for e in entities},
        predicates={r: Predicate.unknown_from_name(r) for _, r, _ in kg.relations},
        entity_descriptions=None,
        triples=kg.relations,
        entity_text_ids={e: [entry_idx << 16] for e in entities},
        triple_text_ids={t: [entry_idx << 16] for t in kg.relations},
        qualifiers=None,
        entity_aliases={
            e: list(kg.entity_clusters[e]) if e in kg.entity_clusters else [e]
            for e in entities
        },
    )

    # Compute the elapsed time
    end_time = time.perf_counter()
    elapsed_time = end_time - start_time

    # Print the results
    result: dict[str, Any] = {}
    result["args"] = {
        "seed": args.seed,
        "dataset": args.dataset,
        "index": entry_idx,
        "labels": entry.labels,
        "llm": llm_identifier,
        "sent_embed": args.sent_embed,
    }
    result["texts"] = entry.texts
    result["graph"] = graph_to_dict(formatted_kg)
    result["usage"] = {"generation": {k: asdict(m) for k, m in usage_info.items()}}
    result["elapsed_time"] = elapsed_time
    with open(result_filepath, "w") as fp:
        json.dump(result, fp)


if __name__ == "__main__":
    args = parser.parse_args()

    # Load the data
    data = load_dataset(args.dataset)

    # Load the sentence embedder
    sent_embed = SentenceTransformer(args.sent_embed)

    # Retrieve the LLM name from the OpenAI client wrapper
    client = OpenAIClient(
        model=args.llm,
        base_url=args.llm_api_base,
        api_key=args.api_key,
        seed=args.seed,
        verbose=args.verbose,
    )
    llm_identifier = client.model

    outputs_path = os.path.join(
        "results",
        "kgc-kggen" if not args.out_alias else f"kgc-kggen-{args.out_alias}",
        args.dataset,
        format_llm_name(llm_identifier),
        build_results_path(
            seed=args.seed,
        ),
    )
    os.makedirs(outputs_path, exist_ok=True)

    # Run the experiments
    indices: list[int] = ranges_to_indices(args.ranges)
    if args.n_jobs == 1:
        for entry_idx in indices:
            run_experiment(entry_idx)
    else:
        thread_map(run_experiment, indices, max_workers=args.n_jobs, desc="Experiments")
