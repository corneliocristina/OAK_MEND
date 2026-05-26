import argparse
import json
import math
import os
import subprocess
import time
from collections import defaultdict
from dataclasses import asdict

import numpy as np
from tqdm import tqdm

from okgc.utils.query_patterns import (
    BGP,
    QUERY_PATTERN_TEMPLATES,
    extract_entities_predicates_from_bgp,
)
from okgc.utils.sparql import sparql_query
from scripts.eval.patterns.qlever_utils import build_qlever_cmd

parser = argparse.ArgumentParser(
    prog="Generate and rank patterns given an ontology graph"
)
parser.add_argument(
    "--rank-predicates-popularity",
    action="store_true",
    default=False,
    help="Whether to rank the generated patterns by the popularity of the predicates in it",
)
parser.add_argument(
    "--port",
    default=7715,
    type=int,
    help="The port of the qlever endpoint where the ontology graph is stored",
)
parser.add_argument(
    "--kg-name",
    help="The KG name to pass to QLever",
    default="ontology-graph",
    type=str,
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


if __name__ == "__main__":
    args = parser.parse_args()

    # Build the inputs path
    inputs_path = os.path.join("kgs", args.kg_name)

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
        "qlever-ontology-kg",
        "--num-threads",
        "4",
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

    # For each query pattern structure, find all the predicates in that pattern
    # as found in the ontology graph
    endpoint = f"http://localhost:{args.port}"
    patterns: dict[str, list[dict[str, str]]] = defaultdict(list)
    patterns_sets: dict[str, set[frozenset[str]]] = defaultdict(set)
    for name, bgp in tqdm(QUERY_PATTERN_TEMPLATES.items(), desc="Pattern structures"):
        _, predicates = extract_entities_predicates_from_bgp(bgp)
        columns = " ".join(sorted([f"?{r}" for r in predicates]))
        query = f"SELECT DISTINCT {columns} WHERE {{ {bgp.body} }}"
        outputs = sparql_query(endpoint=endpoint, query=query)
        bindings = outputs["results"]["bindings"]
        for binding in bindings:
            matched_predicates = {}
            for r, d in binding.items():
                matched_predicates[r] = d["value"].split("/")[-1]
            matched_predicates_set = frozenset(matched_predicates.values())
            # We want to select interesting queries
            # So, we choose query patterns whose predicates are distinct
            if len(matched_predicates_set) != len(matched_predicates):
                continue
            # Remove combinations of predicate patterns that have been already found
            # This is  to reduce the number of patterns that are too similar,
            # and therefore reduce the cost of counting the pattern matches later on
            if matched_predicates_set in patterns_sets[name]:
                continue
            patterns[name].append(matched_predicates)
            patterns_sets[name].add(matched_predicates_set)

    # Stop the QLever endpoint
    qlever_stop_cmd = [
        "qlever",
        "stop",
        "--name",
        f"{args.kg_name}",
        "--server-container",
        "qlever-ontology-kg",
    ]
    qlever_stop_cmd = build_qlever_cmd(
        qlever_stop_cmd,
        singularity=args.singularity,
        qlever_simg_filepath=args.qlever_simg_filepath,
    )
    subprocess.run(qlever_stop_cmd, cwd=inputs_path)

    # Check whether to rank the patterns based on the popularity in them
    # This is necessary because for large query structures (e.g., 3p),
    # the number of query patterns quickly explodes. So, ranking and filtering
    # them (e.g., taking the first 1000 only) reduces computational cost when
    # we later count the query pattern matches
    if args.rank_predicates_popularity:
        # Heuristic: the popularity score of a query pattern with predicates r_1, ..., r_k
        # is equal to \sum_{i=1}^k \log K_i, where K_i is the number of triples in Wikidata
        # having r_i as predicate. Therefore, we rank patterns with popular predicates higher
        #
        # (1) Collect all the predicates found
        predicates = set(
            p for ps in patterns.values() for pd in ps for p in pd.values()
        )
        # (2) For each predicate, query Wikidata to find the number of triples with that predicate
        predicate_freqs: dict[str, int] = {}
        for p in tqdm(predicates, desc="Predicates popularity"):
            query = f"SELECT (COUNT(*) as ?n) WHERE {{ ?x1 wdt:{p} ?x2 }}"
            outputs = sparql_query(query=query)
            bindings = outputs["results"]["bindings"]
            assert len(bindings) == 1, bindings
            predicate_freqs[p] = int(bindings[0]["n"]["value"])
        # (3) Compute the scores of pattern matches
        patterns_scores: dict[str, list[float]] = defaultdict(list)
        for name, ps in patterns.items():
            for pd in ps:
                score = sum(math.log(predicate_freqs[p]) for p in pd.values())
                patterns_scores[name].append(score)
        # (4) Sort all the patterns based on their score (descending)
        patterns_sorted: dict[str, list[dict[str, str]]] = {}
        for name, ps in patterns_scores.items():
            idx = np.argsort(ps).tolist()[::-1]
            patterns_sorted[name] = [patterns[name][i] for i in idx]
        patterns = patterns_sorted

    # Plot some stats
    for name, ps in patterns.items():
        print(f"Pattern structure '{name}' -- number of patterns: {len(ps)}")

    # Serialize the (possibly ranked) patterns
    bgps: dict[str, list[BGP]] = defaultdict(list)
    for name, ps in patterns.items():
        bgp_template = QUERY_PATTERN_TEMPLATES[name]
        for pd in ps:
            bgp_body = str(bgp_template.body)
            for r, p in pd.items():
                bgp_body = bgp_body.replace(f"?{r}", f"wdt:{p}")
            bgp = BGP(bgp_body, bgp_template.num_triples, bgp_template.max_arity)
            bgps[name].append(bgp)
    destination_path = os.path.join("data", "query-bgps")
    os.makedirs(destination_path, exist_ok=True)
    filepath = os.path.join(destination_path, "artificial-patterns.json")
    with open(filepath, "w") as fp:
        formatted_bgps = {
            name: [asdict(bgp) for bgp in ps] for name, ps in bgps.items()
        }
        json.dump(formatted_bgps, fp)
