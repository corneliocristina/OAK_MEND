import argparse
import json
import os
import traceback
from typing import Any

from tqdm.contrib.concurrent import thread_map

from okgc.datasets.loaders import DATASET_NAMES, load_dataset
from okgc.qa.zero_shot_qa import ZeroShotQA
from okgc.utils.filesystem import build_results_path, format_llm_name
from okgc.utils.metrics import evaluate_qa_answer
from okgc.utils.openai_wrapper import OpenAIClient
from scripts.utils import ranges_to_indices

parser = argparse.ArgumentParser(prog="QA for the zero-shot baseline")
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
    "--api-key", help="The API key", default="okgc-reflect-key-deadbeef"
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
        # Evaluate with zero shot QA
        pred_answer = zero_shot_qa.ask(question)
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
        "sent_embed": None,
    }
    result["qa"] = {
        "pred_answer": pred_answer,
        "ground_truth_answer": ground_truth_answer,
        "metrics": evaluate_qa_answer(pred_answer, ground_truth_answer),
    }
    with open(result_filepath, "w") as fp:
        json.dump(result, fp)


if __name__ == "__main__":
    args = parser.parse_args()

    # Load the data
    data = load_dataset(args.dataset)

    # Setup the LLM client
    client = OpenAIClient(
        args.llm,
        base_url=args.llm_api_base,
        api_key=args.api_key,
        seed=args.seed,
        verbose=args.verbose,
    )

    # Builds the paths
    llm_name = format_llm_name(client.model)
    inner_path = build_results_path(
        seed=args.seed,
    )
    outputs_path = os.path.join(
        "results",
        "qa-zero-shot" if not args.alias else f"qa-zero-shot-{args.alias}",
        args.dataset,
        llm_name,
        inner_path,
        f"filter-triplesFalse-qualifiersFalse",
    )
    os.makedirs(outputs_path, exist_ok=True)

    # Retrieve the question-ground truth answer pairs
    qa_pairs = [
        (i, x.extra_info["question"], x.extra_info["answer"])
        for i, x in enumerate(data)
    ]

    # Instantiate the QA object
    zero_shot_qa = ZeroShotQA(client, verbose=args.verbose)

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
