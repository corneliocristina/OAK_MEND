import argparse
import json
import os
import time
import traceback
from dataclasses import asdict
from typing import Any

from tqdm.contrib.concurrent import thread_map

from okgc import OntologyMend
from okgc.datasets.loaders import (
    DATASET_NAMES,
    load_predicates,
    load_qualifiers,
    load_types_hierarchy,
)
from okgc.utils.filesystem import (
    build_results_path,
    format_llm_name,
    load_results,
)
from okgc.utils.graph import dict_to_graph, graph_to_dict
from okgc.utils.openai_wrapper import OpenAIClient
from okgc.utils.sent_embed import SentenceEmbedder
from okgc.utils.vector_index import PredicatesVectorIndex, TypesVectorIndex
from scripts.utils import ranges_to_indices

parser = argparse.ArgumentParser(
    prog="Corrections Processor",
    description="Apply some triple corrections and store the results",
)
parser.add_argument("dataset", choices=DATASET_NAMES)
parser.add_argument(
    "--ranges",
    help="The ranges of the entry indices to process",
    required=True,
    type=str,
)
parser.add_argument(
    "--llm-api-base",
    help="The LLM API url including the port",
    default="http://localhost:9110/v1",
)
parser.add_argument(
    "--llm",
    help="The LLM to use",
    default="",
)
parser.add_argument(
    "--sent-embed-api-base",
    help="The sentence embedding model API url including the port",
    default="http://localhost:9200/v1",
)
parser.add_argument(
    "--sent-embed",
    help="The sentence embedding model to use",
    default="",
)
parser.add_argument(
    "--api-key", help="The API key", default="okgc-reflect-key-deadbeef"
)
parser.add_argument(
    "--think",
    help="Whether to enable thinking/reasoning to speedup generation",
    default=False,
    action="store_true",
)
parser.add_argument(
    "--in-alias", help="The alias of the input results directory", default="", type=str
)
parser.add_argument(
    "--out-alias",
    help="The alias of the output results directory",
    default="",
    type=str,
)
parser.add_argument(
    "--correct-qualifiers",
    help="Whether to also correct the qualifiers",
    default=False,
    action="store_true",
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


def process_result_entry(result: dict[str, Any]):
    entry_idx = result["args"]["index"]
    processed_result_filepath = os.path.join(outputs_path, f"entry-{entry_idx:04}.json")
    # Check if a result has already been serialized
    # Also, check if the result entry index is in the input ranges
    if os.path.isfile(processed_result_filepath) or entry_idx not in indices:
        if args.verbose and entry_idx in indices:
            print(f"Skipping entry index {entry_idx:04} ...")
        return

    # Retrieve the graph
    graph = dict_to_graph(result["graph"])

    # Retrieve the text for the triples in the graph
    texts: dict[int, str] = {
        (entry_idx << 16) + sub_text_id: text
        for sub_text_id, text in enumerate(result["texts"])
    }

    # Instantiate the ontology-guided reflect loop object
    omend = OntologyMend(
        client=client,
        # Set whether to also correct the qualifiers with LLM calls
        correct_qualifiers=args.correct_qualifiers,
        # Set the indexes
        types_index=types_index,
        predicates_index=predicates_index,
        qualifier_predicates_index=qualifier_predicates_index,
        # Set the verbose level
        verbose=args.verbose,
    )

    # Run the corrections loop
    start_time = time.perf_counter()
    try:
        graph, origins, usage_info = omend.apply_repeated(
            graph,
            texts,
        )
    except Exception:
        print(f"Failed entry idx {entry_idx}. Message: {traceback.format_exc()}")
        return

    # Compute the elapsed time
    end_time = time.perf_counter()
    elapsed_time = end_time - start_time

    # Serialize the results
    processed_result: dict[str, Any] = {}
    processed_result["args"] = result["args"]
    processed_result["texts"] = result["texts"]
    processed_result["graph"] = graph_to_dict(graph)
    processed_result["usage"] = result["usage"]
    processed_result["usage"].update({"corrections": asdict(usage_info)})
    processed_result["corrections"] = {
        "origins": [{"triple": t, "origin": origin} for t, origin in origins.items()],
        "transform_counters": {
            "triples": asdict(omend.triples_transform_counters),
            "qualifiers": asdict(omend.qualifiers_transform_counters),
        },
    }
    processed_result["elapsed_time"] = elapsed_time + result["elapsed_time"]

    # Serialize the results
    with open(processed_result_filepath, "w") as fp:
        json.dump(processed_result, fp)


if __name__ == "__main__":
    args = parser.parse_args()

    # Load the predicates and the types
    predicates, inverted_predicates = load_predicates(args.dataset)
    qualifiers, inverted_qualifiers = load_qualifiers(args.dataset)
    types_hierarchy = load_types_hierarchy(args.dataset)

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

    # Initialize the LLM client
    client = OpenAIClient(
        args.llm, base_url=args.llm_api_base, api_key=args.api_key, verbose=args.verbose
    )

    # Initialize the sentence embedder
    sent_embed = SentenceEmbedder(
        client=OpenAIClient(
            args.sent_embed, base_url=args.sent_embed_api_base, api_key=args.api_key
        )
    )

    # Build the paths
    llm_name = format_llm_name(client.model)
    inner_filepath = build_results_path(
        think=args.think,
        seed=args.seed,
    )
    inputs_path = os.path.join(
        "results",
        "kgc-okgc" if not args.in_alias else f"kgc-okgc-{args.in_alias}",
        args.dataset,
        llm_name,
        inner_filepath,
    )
    print(inputs_path)
    outputs_path = os.path.join(
        "results",
        "kgc-oak-mend" if not args.out_alias else f"kgc-oak-mend-{args.out_alias}",
        args.dataset,
        llm_name,
        inner_filepath,
    )
    os.makedirs(outputs_path, exist_ok=True)

    # Load the results
    indices: list[int] = ranges_to_indices(args.ranges)
    results = load_results(inputs_path)

    print("Applying corrections to the generated triples ...")
    thread_map(process_result_entry, results, max_workers=args.n_jobs, desc="Results")
