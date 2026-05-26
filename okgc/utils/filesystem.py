import json
import os
from typing import Any


def format_llm_name(name: str) -> str:
    return name.split("/")[-1].replace(":", "-")


def build_results_path(
    *,
    think: bool | None = False,
    context_predicates: bool | None = None,
    context_domain_range: bool | None = None,
    seed: int | None = None,
) -> str:
    dirs = []
    if think is not None:
        dirs.append(f"think{think}")
    if context_predicates is not None:
        dirs.append(f"ctxp{context_predicates}")
    if context_domain_range is not None:
        dirs.append(f"ctxdr{context_domain_range}")
    if seed is not None:
        dirs.append(f"seed{seed}")
    if not dirs:
        return ""
    return os.path.join(*dirs)


def load_results(path: str, *, verbose: bool = False) -> list[dict[str, Any]]:
    results = []
    for result_filepath in os.listdir(path):
        filepath = os.path.join(path, result_filepath)
        if verbose:
            print(f"Loading results from {filepath}")
        with open(filepath, "r") as fp:
            result = json.load(fp)
        results.append(result)
    return results


def load_results_repetitions(
    path: str, *extra_paths: str, verbose: bool = False
) -> dict[int, list[dict[str, Any]]]:
    results = {}
    for repetition_path in os.listdir(path):
        assert repetition_path[:4] == "seed"
        seed = int(repetition_path[4:])
        root_path = os.path.join(path, repetition_path, *extra_paths)
        if verbose:
            print(f"Loading results from directory {root_path}")
        results[seed] = load_results(root_path, verbose=verbose)
    return results


def load_prompt(filepath: str) -> str:
    with open(filepath, "r") as fp:
        prompt = fp.read()
    return prompt
