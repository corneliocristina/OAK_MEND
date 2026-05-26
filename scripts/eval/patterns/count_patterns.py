import argparse
import json
import os
import subprocess
import time

import numpy as np
from dacite import from_dict
from tqdm import tqdm
from tqdm.contrib.concurrent import thread_map

from okgc.datasets.loaders import DATASET_NAMES, load_predicates
from okgc.utils.filesystem import build_results_path, format_llm_name
from okgc.utils.query_patterns import BGP
from okgc.utils.sparql import WIKIDATA_PROPERTY_DIRECT_PREFIX, sparql_query
from scripts.eval.patterns.qlever_utils import build_qlever_cmd

parser = argparse.ArgumentParser(
    prog="Count the number of patterns in a knowledge graph"
)
parser.add_argument("dataset", choices=DATASET_NAMES)
parser.add_argument(
    "--llm",
    help="The LLM to use, with the provider",
    default="hosted_vllm/Qwen/Qwen3-32B-AWQ",
)
parser.add_argument(
    "--alias",
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
    "--kg-name",
    help="The KG name to pass to QLever",
    default="knowledge-graph",
    type=str,
)
parser.add_argument(
    "--patterns-filepath", required=True, help="The query patterns filepath"
)
parser.add_argument(
    "--patterns-limit",
    default=10000,
    type=int,
    help="The maximum number of query patterns to consider. Use -1 to use all of them",
)
parser.add_argument(
    "--port",
    default=7715,
    type=int,
    help="The port of the qlever endpoint where the knowledge graph is stored",
)
parser.add_argument(
    "--top-k-patterns",
    default=15,
    type=int,
    help="Print the match counts of the top-k patterns",
)
parser.add_argument(
    "--singularity",
    help="Whether to use singularity to run the QLever docker container",
    action="store_true",
    default=False,
)
parser.add_argument(
    "--qlever-simg-filepath",
    help="The filepath to the QLever SIMG image",
    default="",
)
parser.add_argument(
    "--n-jobs", help="The number of jobs for multiprocessing", default=32, type=int
)


def run_bgp_counting(bgp: BGP) -> int:
    query = f"SELECT (COUNT(*) as ?n) WHERE {{ {bgp.body} }}"
    query = f"PREFIX wdt: <{WIKIDATA_PROPERTY_DIRECT_PREFIX}>\n{query}"
    outputs = sparql_query(endpoint=ENDPOINT, query=query)
    bindings = outputs["results"]["bindings"]
    n = int(bindings[0]["n"]["value"])
    return n


if __name__ == "__main__":
    args = parser.parse_args()

    # Build the inputs path
    inner_path = build_results_path(
        seed=args.seed,
    )
    inputs_path = os.path.join(
        "kgs",
        "kgc" if not args.alias else f"kgc-{args.alias}",
        args.dataset,
        format_llm_name(args.llm),
        inner_path,
    )

    # Build the outputs path
    outputs_path = os.path.join(
        "results",
        "qcounts" if not args.alias else f"qcounts-{args.alias}",
        args.dataset,
        format_llm_name(args.llm),
        inner_path,
    )
    os.makedirs(outputs_path, exist_ok=True)

    # Load the predicates associated to the input dataset
    predicates, inverted_predicates = load_predicates(args.dataset)
    predicate_codes = {p.code: p for _, p in predicates.items()}

    # Load the query patterns from file
    with open(args.patterns_filepath) as fp:
        patterns_json = json.load(fp)
    patterns: dict[str, list[BGP]] = {
        name: [from_dict(BGP, data=bgp) for bgp in bgps]
        for name, bgps in patterns_json.items()
    }
    patterns_identifier = (
        "lsq" if len(patterns) == 1 and "lsq" in patterns else "artificial"
    )

    # Start the QLever endpoint
    qlever_start_cmd = [
        "qlever",
        "start",
        "--name",
        f"{args.kg_name}",
        "--description",
        "empty",
        "--port",
        f"{args.port}",
        "--kill-existing-with-same-port",
        "--cache-max-size",
        "64GB",
        "--memory-for-queries",
        "64GB",
        "--cache-max-size-single-entry",
        "16GB",
        "--server-container",
        "qlever-kg",
        "--num-threads",
        f"{args.n_jobs}",
        "--timeout",
        "600s",
    ]
    qlever_start_cmd = build_qlever_cmd(
        qlever_start_cmd,
        singularity=args.singularity,
        qlever_simg_filepath=args.qlever_simg_filepath,
    )
    qlever_process = subprocess.Popen(qlever_start_cmd, cwd=inputs_path)
    time.sleep(15)
    qlever_status = qlever_process.poll()
    assert qlever_status is None, (
        f"qlever start has crashed: return code={qlever_status}"
    )

    # For each query pattern structure and for each pattern,
    # obtain the number of matches in the knowledge graph
    ENDPOINT = f"http://localhost:{args.port}"
    pattern_counts: dict[str, list[int]] = {}
    for name, ps in tqdm(patterns.items(), "BGP structures", position=0, leave=True):
        if args.patterns_limit > 0:
            random_state = np.random.RandomState(42)
            random_state.shuffle(ps)
            ps = ps[: args.patterns_limit]
        counts = thread_map(
            run_bgp_counting,
            ps,
            max_workers=args.n_jobs,
            position=1,
            leave=False,
            desc=f"[{name}]",
        )
        pattern_counts[name] = counts

    # Stop the QLever endpoint
    qlever_stop_cmd = [
        "qlever",
        "stop",
        "--name",
        f"{args.kg_name}",
        "--server-container",
        "qlever-kg",
    ]
    qlever_stop_cmd = build_qlever_cmd(
        qlever_stop_cmd,
        singularity=args.singularity,
        qlever_simg_filepath=args.qlever_simg_filepath,
    )
    subprocess.run(qlever_stop_cmd, cwd=inputs_path)

    # Serialize the results
    results: dict[str, dict] = {}
    for name, counts in pattern_counts.items():
        results[name] = {
            "num_patterns": len(counts),
            "num_matches": sum(counts),
            "counts": counts,
        }
    filepath = os.path.join(outputs_path, f"{patterns_identifier}_patterns_counts.json")
    with open(filepath, "w") as fp:
        json.dump(results, fp)
