import argparse
import json
import os
import traceback
from typing import Any

from tqdm.contrib.concurrent import thread_map

from okgc.datasets.loaders import (
    DATASET_NAMES,
    load_dataset,
    load_predicates,
    load_types_hierarchy,
)
from okgc.qa.multi_step_qa import MultiStepQA
from okgc.utils.filesystem import build_results_path, format_llm_name, load_results
from okgc.utils.graph import dict_to_graph, merge_graphs
from okgc.utils.metrics import evaluate_qa_answer
from okgc.utils.openai_wrapper import OpenAIClient
from okgc.utils.sent_embed import SentenceEmbedder
from scripts.utils import ranges_to_indices

parser = argparse.ArgumentParser(
    prog="QA evaluation launcher", description="Evaluate the generated KGs based on QA"
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
    "--api-key", help="The API key", default="okgc-reflect-key-deadbeef"
)
parser.add_argument(
    "--no-qualifiers",
    help="Whether to remove all qualifiers before performing QA",
    action="store_true",
    default=False,
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
    "--no-index-qualifiers-entities",
    help="Whether to NOT index the qualifier entities for QA",
    action="store_true",
    default=False,
)
parser.add_argument(
    "--alias",
    help="The alias of the input/output results directory",
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


def run_evaluation(qa_pair: tuple[int, str, str]):
    # Get question and ground truth
    entry_index, question, ground_truth_answer = qa_pair

    # Build the result filepath string
    result_filepath = os.path.join(outputs_path, f"entry-{entry_index:04}.json")
    if os.path.isfile(result_filepath):
        print(f"Skipping entry index {entry_index:04} ...")
        return

    try:
        # Evaluate with multi step QA
        pred_answer = multi_step_qa.ask(question)
    except Exception:
        print(
            f"Failed QA pair\nQuestion: {question}\nGround Truth Answer: {ground_truth_answer}\nMessage:\n{traceback.format_exc()}"
        )
        return

    # Print the results
    result: dict[str, Any] = {}
    result["args"] = {
        "seed": args.seed,
        "index": entry_index,
        "dataset": args.dataset,
        "llm": client.model,
        "sent_embed": sent_embed.client.model,
        "no_qualifiers": args.no_qualifiers,
        "filter_triples": args.filter_triples,
        "filter_qualifiers": args.filter_qualifiers,
        "no_index_qualifiers_entities": args.no_index_qualifiers_entities,
    }
    result["qa"] = {
        "pred_answer": pred_answer,
        "ground_truth_answer": ground_truth_answer,
        "metrics": evaluate_qa_answer(
            pred_answer, ground_truth_answer, aliases=kg.entity_aliases
        ),
    }
    with open(result_filepath, "w") as fp:
        json.dump(result, fp)


if __name__ == "__main__":
    args = parser.parse_args()

    # Load the data, the predicates and the types
    data = load_dataset(args.dataset)
    predicates, inverted_predicates = load_predicates(args.dataset)
    types_hierarchy = load_types_hierarchy(args.dataset)

    # Setup the LLM client and the sentence emedder
    client = OpenAIClient(
        args.llm,
        base_url=args.llm_api_base,
        api_key=args.api_key,
        seed=args.seed,
        verbose=args.verbose,
    )
    sent_embed = SentenceEmbedder(
        client=OpenAIClient(
            args.sent_embed, base_url=args.sent_embed_api_base, api_key=args.api_key
        )
    )

    # Builds the paths
    llm_name = format_llm_name(client.model)
    inner_path = build_results_path(
        seed=args.seed,
    )
    inputs_path = os.path.join(
        "results",
        "kgc" if not args.alias else f"kgc-{args.alias}",
        args.dataset,
        llm_name,
        inner_path,
    )
    out_path = "qa" if not args.alias else f"qa-{args.alias}"
    if args.no_qualifiers:
        out_path += "-no-qualifiers"
    if args.no_index_qualifiers_entities:
        out_path += "-no-index-qs-entities"
    outputs_path = os.path.join(
        "results",
        out_path,
        args.dataset,
        llm_name,
        inner_path,
        f"filter-triples{args.filter_triples}-qualifiers{args.filter_qualifiers}",
    )
    os.makedirs(outputs_path, exist_ok=True)

    # Load the results
    results = load_results(inputs_path)

    # Load and merge the knowledge graph fragments into a single one
    fragments = [dict_to_graph(r["graph"]) for r in results]
    kg = merge_graphs(fragments)

    # Retrieve the question-ground truth answer pairs
    entry_indices = set(r["args"]["index"] for r in results)
    qa_pairs = [
        (i, x.extra_info["question"], x.extra_info["answer"])
        for i, x in enumerate(data)
        if i in entry_indices
    ]

    # Instantiate the QA object
    multi_step_qa = MultiStepQA(
        client,
        sent_embed,
        kg,
        use_qualifiers=not args.no_qualifiers,
        filter_triples=args.filter_triples,
        filter_qualifiers=args.filter_qualifiers,
        index_qualifiers_entities=not args.no_index_qualifiers_entities,
        types_hierarchy=types_hierarchy,
        predicates=predicates,
        verbose=args.verbose,
    )

    # Run the evaluations
    indices: list[int] = ranges_to_indices(args.ranges)
    qa_pairs = [(i, q, a) for (i, q, a) in qa_pairs if i in indices]
    if args.n_jobs == 1:
        for x in qa_pairs:
            run_evaluation(x)
    else:
        thread_map(
            run_evaluation, qa_pairs, max_workers=args.n_jobs, desc="Evaluations"
        )
