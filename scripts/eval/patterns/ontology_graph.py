import argparse
import itertools
import json
import os
import subprocess
from collections import defaultdict

import rdflib
from tqdm import tqdm

from okgc.datasets.loaders import load_predicates, load_types_hierarchy
from okgc.utils.sparql import TypeInfo
from scripts.eval.patterns.qlever_utils import build_qlever_cmd

parser = argparse.ArgumentParser(
    prog="Save the ontology graph to file",
    description="An ontology graph is a KG where the entities are types and the links denote domain-range information",
)
parser.add_argument(
    "--types-minimum-degree",
    default=1,
    type=int,
    help="The minimum in/out-degree of the type entities in the graph",
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

    # Build the output directory
    outputs_path = os.path.join("kgs", args.kg_name)
    os.makedirs(outputs_path, exist_ok=True)

    # Load the predicates and the types
    predicates, inverted_predicates = load_predicates("hotpot1000")
    types_hierarchy = load_types_hierarchy("hotpot1000")
    types = types_hierarchy.types

    # Load the predicates specifically used to generate the patterns
    # These predicates have been obtained as the union of common predicates
    # appearing in ReDocRED and CodRED datasets
    with open(os.path.join("data", "wikidata", "predicates4patterns.json"), "r") as fp:
        pattern_predicates = json.load(fp)

    # Remove predicates that do not have domain-range information
    # and that do not appear in the list of predicates for generating the patterns
    predicates = {
        name: p
        for name, p in predicates.items()
        if p.code in pattern_predicates and (len(p.domain) > 0 or len(p.range) > 0)
    }

    # Compute the in/out-degree of the types
    type_degrees: dict[str, int] = defaultdict(int)
    for p in predicates.values():
        for c in p.domain:
            type_degrees[c] += 1
        for c in p.range:
            type_degrees[c] += 1
    # Filter out types that do not appear often enough in the domain and range of predicates
    # That is, we remove rare entity types. This makes the generation and counting of query patterns
    # more efficient, while retaining common query patterns
    type_codes: dict[str, TypeInfo] = {
        c: types[c]
        for c, degree in type_degrees.items()
        if degree >= args.types_minimum_degree
    }

    # The prefixes for predicates
    prefix_predicates = "http://www.wikidata.org/prop/direct/"
    # Construct the URIs for predicates and types
    type_uris = {c: rdflib.URIRef(f"{c}") for c in type_codes}
    predicate_uris = {
        p.code: rdflib.URIRef(f"{p.code}", base=prefix_predicates)
        for p in predicates.values()
    }

    # Instantiate an empty graph and bind the namespaces
    graph = rdflib.Graph()
    graph.bind("wdt", rdflib.Namespace(prefix_predicates))

    # Note that for literals we will make use of dummy type URIs
    literal_dummy_uri_id = 0

    def next_dummy_uri() -> rdflib.URIRef:
        global literal_dummy_uri_id
        uri = rdflib.URIRef(f"D{literal_dummy_uri_id}")
        literal_dummy_uri_id += 1
        return uri

    # Construct the ontology graph
    for p in tqdm(predicates.values(), desc="Predicates"):
        # Retrieve the domain types URIs, if any
        domain_uris: list[rdflib.URIRef]
        if p.domain:
            domain_uris = []
            for c in p.domain:
                if c not in type_uris:
                    continue
                domain_uris.append(type_uris[c])
        else:
            domain_uris = [next_dummy_uri()]
        # Retrieve the range types URIs, if any
        range_uris: list[rdflib.URIRef]
        if p.range:
            range_uris = []
            for c in p.range:
                if c not in type_uris:
                    continue
                range_uris.append(type_uris[c])
        else:
            dummy_uri = rdflib.URIRef(f"dummy{literal_dummy_uri_id}")
            range_uris = [next_dummy_uri()]
        # Get the predicate URI
        r_uri = predicate_uris[p.code]
        # Add links connecting all domain types with all range types
        for s_uri, o_uri in itertools.product(domain_uris, range_uris):
            graph.add((s_uri, r_uri, o_uri))

    # Print some stats
    unique_predicates = set()
    for s, r, o in graph:
        unique_predicates.add(r)
    print(f"Minimum degree for types: {args.types_minimum_degree}")
    print(f"Number of triples: {len(graph)}")
    print(f"Unique predicates: {len(unique_predicates)}")

    # Serialize the ontology graph
    kg_filename = "kg.ttl"
    filepath = os.path.join(outputs_path, kg_filename)
    graph.serialize(filepath)

    # Build the index with QLever
    qlever_index_cmd = [
        "qlever",
        "index",
        "--name",
        f"{args.kg_name}",
        "--overwrite",
        "--input-files",
        f"{kg_filename}",
        "--cat-input-files",
        f"cat {kg_filename}",
    ]
    qlever_index_cmd = build_qlever_cmd(
        qlever_index_cmd,
        singularity=args.singularity,
        qlever_simg_filepath=args.qlever_simg_filepath,
    )
    subprocess.run(qlever_index_cmd, cwd=outputs_path)
