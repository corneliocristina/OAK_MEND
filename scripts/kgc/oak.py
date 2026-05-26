import argparse
import json
import os
import time
import traceback
from collections import defaultdict
from dataclasses import asdict
from typing import Any

from tqdm.contrib.concurrent import thread_map

from okgc import OAK
from okgc.datasets.loaders import (
    DATASET_NAMES,
    load_dataset,
    load_predicates,
    load_qualifiers,
    load_types_hierarchy,
)
from okgc.utils.filesystem import build_results_path, format_llm_name
from okgc.utils.graph import (
    Graph,
    graph_to_dict,
    merge_graphs,
)
from okgc.utils.openai_wrapper import OpenAIClient
from okgc.utils.sent_embed import SentenceEmbedder
from okgc.utils.usage import UsageInfo
from okgc.utils.vector_index import PredicatesVectorIndex, TypesVectorIndex
from scripts.utils import ranges_to_indices

parser = argparse.ArgumentParser(
    prog="Experiment OAK Launcher",
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
    "--sent-embed-api-base",
    help="The sentence embedding model API url including the port",
    default="http://localhost:9200/v1",
)
parser.add_argument(
    "--sent-embed",
    help="The sentence embedding model to use. If empty, then it defaults to the first model available on the endpoint",
    default="",
)
parser.add_argument(
    "--dedup-entities-method",
    help="The method used to deduplicate the entities after KG merging",
    choices=["spacy", "llm"],
    default="llm",
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

    # Generate the knowledge graph
    start_time = time.perf_counter()
    kgs: list[Graph] = []
    usage_info: dict[str, UsageInfo] = defaultdict(UsageInfo)
    assert len(entry.texts) < (2**16)
    try:
        for sub_text_id, text in enumerate(entry.texts):
            text_id = (entry_idx << 16) + sub_text_id
            kg, gen_usage = oak.generate_from_text(text, text_id=text_id)
            kgs.append(kg)
            for k, u in gen_usage.items():
                usage_info[k] += u
        kg = merge_graphs(kgs)
        kg, dedup_usage = oak.deduplicate_entities(
            kg, method=args.dedup_entities_method
        )
        usage_info["deduplicate_entities"] = dedup_usage
    except Exception:
        print(f"Failed entry index {entry_idx}. Message:\n{traceback.format_exc()}")
        return

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
        "llm": client.model,
        "sent_embed": sent_embed.client.model,
    }
    result["texts"] = entry.texts
    result["graph"] = graph_to_dict(kg)
    result["usage"] = {"generation": {k: asdict(m) for k, m in usage_info.items()}}
    result["elapsed_time"] = elapsed_time
    with open(result_filepath, "w") as fp:
        json.dump(result, fp)


if __name__ == "__main__":
    args = parser.parse_args()

    # Load the data, the predicates and the types
    data = load_dataset(args.dataset)
    predicates, inverted_predicates = load_predicates(args.dataset)
    qualifiers, inverted_qualifiers = load_qualifiers(args.dataset)
    types_hierarchy = load_types_hierarchy(args.dataset)

    # Initialize the OpenAI client wrapper
    client = OpenAIClient(
        model=args.llm,
        base_url=args.llm_api_base,
        api_key=args.api_key,
        seed=args.seed,
        verbose=args.verbose,
    )

    # Initialize indexes
    sent_embed = SentenceEmbedder(
        client=OpenAIClient(
            args.sent_embed, base_url=args.sent_embed_api_base, api_key=args.api_key
        )
    )
    types_index = TypesVectorIndex(
        types_hierarchy, "types_index.pt", sent_embed=sent_embed
    )
    predicates_index = PredicatesVectorIndex(
        predicates, "predicates_index.pt", sent_embed=sent_embed
    )
    qualifier_predicates_index = PredicatesVectorIndex(
        qualifiers, "qualifiers_index.pt", sent_embed=sent_embed
    )

    # Setup the outputs path
    outputs_path = os.path.join(
        "results",
        "kgc-oak" if not args.out_alias else f"kgc-oak-{args.out_alias}",
        args.dataset,
        format_llm_name(client.model),
        build_results_path(
            seed=args.seed,
        ),
    )
    os.makedirs(outputs_path, exist_ok=True)

    # Initialize the OAK method object
    oak = OAK(
        client,
        sent_embed,
        types_index=types_index,
        predicates_index=predicates_index,
        qualifier_predicates_index=qualifier_predicates_index,
        verbose=args.verbose,
    )

    # Run the experiments
    indices: list[int] = ranges_to_indices(args.ranges)
    if args.n_jobs == 1:
        for entry_idx in indices:
            run_experiment(entry_idx)
    else:
        thread_map(run_experiment, indices, max_workers=args.n_jobs, desc="Experiments")
