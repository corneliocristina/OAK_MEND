#!/bin/bash

print_all_results() {
  dataset="$1"
  llm="$2"
  md_results_path="results/mdfiles/$dataset/$llm"
  json_results_path="results/jsonfiles/$dataset/$llm"

  mkdir -p "${md_results_path}"
  mkdir -p "${json_results_path}"

  python -m scripts.eval.kgc_results "${dataset}" --llm "${llm}" --in-alias kggen \
    --md-path "${md_results_path}" --json-path "${json_results_path}"
  python -m scripts.eval.kgc_results "${dataset}" --llm "${llm}" --in-alias wikontic \
    --md-path "${md_results_path}" --json-path "${json_results_path}"
  python -m scripts.eval.kgc_results "${dataset}" --llm "${llm}" --in-alias oak \
    --md-path "${md_results_path}" --json-path "${json_results_path}"
  python -m scripts.eval.kgc_results "${dataset}" --llm "${llm}" --in-alias oak-mend \
    --md-path "${md_results_path}" --json-path "${json_results_path}"
  python -m scripts.eval.kgc_results "${dataset}" --llm "${llm}" --in-alias oak-mend-qualifiers \
    --md-path "${md_results_path}" --json-path "${json_results_path}"
  python -m scripts.eval.kgc_results "${dataset}" --llm "${llm}" --in-alias inctx \
    --md-path "${md_results_path}" --json-path "${json_results_path}"

  python -m scripts.eval.qa.qa_results "${dataset}" --llm "${llm}" --alias zero-shot \
    --md-path "${md_results_path}" --json-path "${json_results_path}"
  python -m scripts.eval.qa.qa_results "${dataset}" --llm "${llm}" --alias vector-rag \
    --md-path "${md_results_path}" --json-path "${json_results_path}"
  python -m scripts.eval.qa.qa_results "${dataset}" --llm "${llm}" --alias kggen \
    --md-path "${md_results_path}" --json-path "${json_results_path}"
  python -m scripts.eval.qa.qa_results "${dataset}" --llm "${llm}" --alias wikontic \
    --md-path "${md_results_path}" --json-path "${json_results_path}"
  python -m scripts.eval.qa.qa_results "${dataset}" --llm "${llm}" --alias wikontic-no-qualifiers \
    --md-path "${md_results_path}" --json-path "${json_results_path}"
  python -m scripts.eval.qa.qa_results "${dataset}" --llm "${llm}" --alias oak \
    --md-path "${md_results_path}" --json-path "${json_results_path}"
  python -m scripts.eval.qa.qa_results "${dataset}" --llm "${llm}" --alias oak-mend-qualifiers \
    --md-path "${md_results_path}" --json-path "${json_results_path}"
  python -m scripts.eval.qa.qa_results "${dataset}" --llm "${llm}" --alias oak-mend-qualifiers-no-qualifiers \
    --md-path "${md_results_path}" --json-path "${json_results_path}"
  python -m scripts.eval.qa.qa_results "${dataset}" --llm "${llm}" --alias inctx \
    --md-path "${md_results_path}" --json-path "${json_results_path}"
  python -m scripts.eval.qa.qa_results "${dataset}" --llm "${llm}" --alias inctx-no-qualifiers \
    --md-path "${md_results_path}" --json-path "${json_results_path}"

  python -m scripts.plots.hscores_query_patterns "${dataset}" \
    --aliases "wikontic;inctx;oak;oak-mend-qualifiers" \
    --labels "Wikontic;InContext;OAK;OAK+MEND" \
    --llm "${llm}" \
    --md-path "${md_results_path}" \
    --json-path "${json_results_path}"
}


print_all_results hotpot1000 Qwen3-30B-A3B-Instruct-2507-FP8
print_all_results hotpot1000 gpt-oss-120b
print_all_results musique1000 Qwen3-30B-A3B-Instruct-2507-FP8
print_all_results musique1000 gpt-oss-120b

python -m scripts.plots.tradeoffs "hotpot1000;musique1000" \
  "Qwen3-30B-A3B-Instruct-2507-FP8;gpt-oss-120b" \
  "kggen;wikontic;inctx;oak-mend-qualifiers" \
  --alias-labels "KGGen;Wikontic;InContext;OAK+MEND" \
  --dataset-labels "HotpotQA;MuSiQue" \
  --json-path results/jsonfiles/
