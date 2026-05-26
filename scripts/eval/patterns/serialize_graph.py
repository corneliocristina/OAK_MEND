import argparse
import os
import subprocess

import rdflib
from tqdm import tqdm

from okgc.datasets.loaders import (
    DATASET_NAMES,
    load_dataset,
    load_predicates,
    load_types_hierarchy,
)
from okgc.utils.filesystem import build_results_path, format_llm_name, load_results
from okgc.utils.graph import dict_to_graph, merge_graphs
from okgc.utils.sparql import TypeInfo
from okgc.utils.typecheck import ontology_filter_triples
from scripts.eval.patterns.qlever_utils import build_qlever_cmd

parser = argparse.ArgumentParser(
    prog="Merge knowledge graph fragments, serialize them into a single knowledge graph and build the index with QLever",
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
    "--alias",
    help="The alias of the results directory",
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
    "--merge-entities",
    help="Whether to merge the entities when merging the KGs in a single one",
    action="store_true",
    default=False,
)
parser.add_argument(
    "--kg-name",
    help="The KG name to pass to QLever",
    default="knowledge-graph",
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

    # Load the KGC results
    inner_path = build_results_path(
        seed=args.seed,
    )
    inputs_path = os.path.join(
        "results",
        "kgc" if not args.alias else f"kgc-{args.alias}",
        args.dataset,
        format_llm_name(args.llm),
        inner_path,
    )
    results = load_results(inputs_path)

    # Build the outputs path
    outputs_path = os.path.join(
        "kgs",
        "kgc" if not args.alias else f"kgc-{args.alias}",
        args.dataset,
        format_llm_name(args.llm),
        inner_path,
    )
    os.makedirs(outputs_path, exist_ok=True)

    # Load the data, the predicates and the types
    data = load_dataset(args.dataset)
    predicates = load_predicates(args.dataset)
    types_hierarchy = load_types_hierarchy(args.dataset)

    # Merge the knowledge graphs
    fragments = []
    for result in tqdm(results, desc="KG fragments"):
        kg = dict_to_graph(result["graph"])
        # Filter out invalid triples
        kg = ontology_filter_triples(kg, hierarchy=types_hierarchy)
        fragments.append(kg)
    kg = merge_graphs(fragments, duplicate_entities=not args.merge_entities)

    # Instantiate an empty graph and bind the predicates namespace
    graph = rdflib.Graph()
    prefix_predicates = "http://www.wikidata.org/prop/direct/"
    graph.bind("wdt", rdflib.Namespace(prefix_predicates))

    # Construct the entities and predictes URIs
    entities_uri = {e: rdflib.URIRef(f"E{i}") for i, e in enumerate(kg.entities)}
    predicates_uri = {
        r: rdflib.URIRef(
            p.code if not p.is_unknown() else p.code.replace(" ", "_"),
            base=prefix_predicates,
        )
        for r, p in kg.predicates.items()
    }
    instance_of_uri = rdflib.URIRef("P31", base=prefix_predicates)

    # Construct the entity types URIs
    types_uri = {}
    for e, ts in kg.entity_types.items():
        for t in ts:
            if t.is_unknown():
                continue
            if t.code in types_uri:
                continue
            types_uri[t.code] = rdflib.URIRef(t.code)

    # Add the triples to the graph
    for s, r, o in kg.triples:
        graph.add((entities_uri[s], predicates_uri[r], entities_uri[o]))
    # Add type information triples
    for e, ts in kg.entity_types.items():
        # Filter-out redundant types given the types hierarchy
        filtered_types: list[TypeInfo] = []
        for t in ts:
            if t.is_unknown():
                continue
            if any(types_hierarchy.is_subclass(u, t) for u in filtered_types):
                continue
            filtered_types = [
                u for u in filtered_types if not types_hierarchy.is_subclass(t, u)
            ]
            filtered_types.append(t)
        for t in filtered_types:
            graph.add((entities_uri[e], instance_of_uri, types_uri[t.code]))

    # Serialize the knowledge graph
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
