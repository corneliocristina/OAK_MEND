[![License: MIT](https://img.shields.io/badge/Code%20License-MIT-green.svg)](https://opensource.org/licenses/MIT)

# OAK + MEND: Knowledge Graph Construction under Ontology Constraints

This repository contains the implementation for the paper _``Better Later Than Sooner: Neuro-Symbolic Knowledge Graph Construction via Ontology-grounded Post-extraction Correction``_.

## Repository Structure

```
.
├── data
│   ├── hotpot1000      # HotpotQA dataset (1000 samples)
│   ├── musique1000     # MuSiQue dataset (1000 samples)
│   ├── wikidata        # Wikidata predicates, types, constraints, ...
│   ├── wikontic-ontology-mappings  # Wikidata mappings in a format for Wikontic only
│   └── query-bgps.zip  # Compressed file containing the two BGPs datasets
├── okgc
│   ├── _oak.py           # Our code for the KG construction
│   ├── _mend.py          # Our code for the KG ongology correction
│   ├── datasets          # Code related to loading the data
│   ├── utils             # General-purpose utilities
│   ├── in_context        # In-context constraints baseline code
│   ├── kg_gen            # KGGen baseline code
│   ├── wikontic          # Wikontic baseline code
│   └── qa
│       ├── multi_step_qa.py  # Multi-step QA as implemented in Wikontic
│       ├── vector_rag_qa.py  # Code for the vector RAG baseline
│       └── zero_shot_qa.py   # Zero-shot QA baseline
├── prompts  # Directory containing the prompts
├── scripts
│   ├── eval         # Scripts for evaluation (consistency, QA, patterns)
│   ├── kgc          # Scripts for KG construction and corrections
│   ├── plots        # Scripts for the plots and for printing the results
│   ├── utils.py     # Utilities for the scripts only
│   └── wdcrawl.py   # Script to retrieve the Wikidata info about predicates, types, constraints, ...
├── shell            # Useful shell scripts
├── kgs              # Destination directory for generated KGs in .ttl format
├── results          # Destination directory for the results
└── vindex           # Destination directory for the vector embedding indexes
```

## Setup and dependencies

We recommend using [uv](https://docs.astral.sh/uv/getting-started/installation/) to manage the Python virtual environment.
Setup the virtual environment and install the dependencies by running:

```shell
uv venv && source .venv/bin/activate
uv pip install -e .
uv pip install pip
python -m spacy download en_core_web_sm
```

## Local LLMs serving

We currently use vLLM to serve both LLMs and embedding models.
For example, to start an OpenAI API-compatible server hosting the ```openai/gpt-oss-120b``` model on ```cuda:0``` and on the localhost port ```9110```, you can run the following:

```shell
CUDA_VISIBLE_DEVICES=0 ./shell/host_vllm.sh openai/gpt-oss-120b 9110
```

Similarly, you can start the sentence embedder ```Qwen/Qwen3-Embedding-0.6B``` on ```cuda:1``` and on the localhost port ```9200``` by running:

```shell
CUDA_VISIBLE_DEVICES=1 ./shell/host_vllm_embedder.sh Qwen/Qwen3-Embedding-0.6B 9200
```

Starting both servers is necessary to run the experiments on KG construction and correction, as well as to run the Wikontic baseline.
You can run multiple instances of the LLMs on different ports and GPUs to parallelize the experiments, or instead use the ```--n-jobs``` argument in most scripts to run multiple experiments in parallel using the same LLM endpoint (see below).

## Generate the knowledge graphs from text

To run our KG construction method (which we call OAK) as well as the baselines locally, you should firstly setup endpoints for the LLM and sentence embedder models as mentioned above.

### OAK

To run OAK on the HotpotQA fragment in ```data/``` and save the generated KGs you can run:

```shell
python -m scripts.kgc.oak hotpot1000 --ranges "0-999" \
  --llm-api-base "http://localhost:9110/v1" --sent-embed-api-base "http://localhost:9200/v1" --verbose
```

As specified by the required argument ```--ranges "0-999"```, this will run the KG construction method on all the 1000 text fragments in HotpotQA, but you can choose which HotpotQA fragments to execute the method on.
The rest of the arguments being shown are optional and can be used to change the LLM and sentence embedder models, as well as the endpoint ports.

> [!NOTE]
> **Where are the generated KGs stored?**
> You can find JSON files storing all the results in the ```results/kgc-oak``` directory.
> If an ```--out-alias {exp alias}``` argument is additionally specified in the KG construction script shown above, then the results will be stored in ```results/kgc-oak-{exp alias}```.

### Running the baselines

Below we give instructions to run some baselines.

#### Wikontic

To run the original implementation of [Wikontic](https://arxiv.org/abs/2512.00590), you need to first setup and run a MongoDB database as follows (requires ```docker``` being installed):

```shell
docker pull mongodb/mongodb-atlas-local:latest
docker run --name wikontic-mongo -p 27018:27017 mongodb/mongodb-atlas-local:latest
```

Then, you need to load part of the Wikidata ontology in MongoDB.
You can do so by (i) setting up the sentence embedder endpoint as shown above, and (ii) by running the following (might take some minutes, done just once):

```shell
python -m okgc.wikontic.create_ontological_triplets_db
python -m okgc.wikontic.create_wikidata_ontology_db \
  --sent-embed-api-base "http://localhost:9200/v1" \
  --mappings_dir ./data/wikontic-ontology-mappings/
```

Finally, you can run Wikontic on all splits of HotpotQA by running:

```shell
python -m scripts.kgc.wikontic hotpot1000 --ranges "0-999" \
  --llm-api-base "http://localhost:9110/v1" --sent-embed-api-base "http://localhost:9200/v1"
```

The KGs generated with Wikontic will be saved to ```results/kgc-wikontic``` by default.

#### KGGen

To run the original implementation of [KGGen](https://arxiv.org/abs/2502.09956), you can run the following:

```shell
python -m scripts.kgc.kggen hotpot1000 --ranges "0-999" \
  --llm-api-base "http://localhost:9110/v1" --verbose
```

The KGs generated with KGGen will be saved to ```results/kgc-kggen``` by default.

#### In-Context Constraints Baseline

To run our custom-built baseline where ontology constraints are added as part of the LLM context upon KG extraction, you can run the following:

```shell
python -m scripts.kgc.in_context hotpot1000 --ranges "0-999" \
  --llm-api-base "http://localhost:9110/v1" --sent-embed-api-base "http://localhost:9200/v1" --verbose
```

The KGs generated with KGGen will be saved to ```results/kgc-inctx``` by default.

## Evaluate the ontology consistency

Once the KGs have been generated, you can evaluate them based on a number of metrics including ontology consistency, e.g., the fraction of triples or qualifiers that are consistent with the ontology constraints.
For this purpose, you can run the following (can take a few minutes based on KG size):

```shell
python -m scripts.eval.kgc_results hotpot1000 --llm openai/gpt-oss-120b --in-alias {exp alias}
```

This will print all the metrics in the terminal.
You can change the experiment alias specified in ```--in-alias {exp alias}``` depending on the method (and other aliases) you used for generating the KGs.
For instance, if you used OAK then you need to specify ```--in-alias oak```, while if you used Wikontic then you need to specify ```--in-alias wikontic```.

## Correcting the knowledge graphs

To run our KG correction method you need to execute the following

```shell
python -m scripts.kgc.mend hotpot1000 --ranges "0-999" --correct-qualifiers \
  --llm-api-base "http://localhost:9110/v1" --sent-embed-api-base "http://localhost:9200/v1"
```

By default, the script will load the KGs from ```results/kgc-oak```.
However, you can load the KGs from ```results/kgc-oak-{exp alias}``` by specifying the flag ```--in-alias {exp alias}``` in the KG correction script above.

Furthermore, you can optionally disable the qualifiers correction step by removing the ```--correct-qualifiers``` flag.
Similarly to the ```scripts.kgc.oak``` script, you can use ```--n-jobs``` to spawn multiple parallel processes using the same LLM and embedding endpoints.
Finally, you can evaluate the ontology consistency of KG after the corrections by using the ```scripts.eval.kgc_results``` script as described above.

> [!NOTE]
> **Where are the corrected KGs stored?**
> You can find JSON files storing all the results in the ```results/kgc-oak-mend``` directory.
> If an ```--out-alias {exp alias}``` argument is additionally specified in the KG construction script above, then the results will be stored in ```results/kgc-oak-mend-{exp alias}```.

## Evaluate on question answering

We use a multi-step algorithm for question answering originally devised for [Wikontic](https://www.arxiv.org/abs/2512.00590).
To evaluate the generated KGs on the task of question answering of HotpotQA, you can run the following:

```shell
python -m scripts.eval.qa.multi_step_qa hotpot1000 --ranges "0-999" \
  --llm-api-base "http://localhost:9110/v1" --sent-embed-api-base "http://localhost:9200/v1" \
  --alias {exp alias} --verbose
```

Note that you need to specify the experiment alias, e.g., ```--alias oak``` for the KGs generated with OAK or ```--alias wikontic``` for KGs generated with Wikontic.
Moreover, similarly to the script used to generate the KGs above, you can specify ```--n-jobs``` to run experiments in parallel.
Finally, the results are stored in ```results/qa-{exp alias}```, and you can print them by running:

```shell
python -m scripts.eval.qa.qa_results hotpot1000 --llm openai/gpt-oss-120b --alias {exp alias}
```

Furthermore, in the above two scripts you can enable the ```--filter-triples``` flag to remove triples violating the ontology constraints before executing the multi-step question answering algorithm.
Similarly, you can enable the both ```--filter-triples``` and ```--filter-qualifiers``` to additionally remove qualifiers violating the ontology constraints.

## Evaluate the number of query patterns

Here, we describe how to count the number of query patterns in the generated KGs, as a higher number of query patterns would translate to a higher number of SPARQL queries returning relevant information.
You can unzip the already-constructed query patterns by running:

```shell
unzip data/query-bgps.zip -d data/
```

Alternatively, you can build the two datasets from scratch as follows.

### Retrieve real-world patterns from the LSQ-2.0 query logs dataset

We extract patterns from the [LSQ-2.0 query logs dataset](https://www.semantic-web-journal.net/content/lsq-20-linked-dataset-sparql-query-logs), which contains queries executed using the Wikidata endpoint.
To retrieve and preprocess the patterns, we need to execute the following:

```shell
python -m scripts.eval.patterns.retrieve_lsq_patterns
```

This will save the filtered and preprocessed patterns in ```data/query-bgps/lsq-bgp-patterns.json```.
Next, we generate the artificial patterns split.

### Generate the artificial query patterns

To generate the artificial query patterns we firstly create a special KG---which we call ontology graph---where nodes are entity types and a directed edge between two types $t_1$ and $t_2$ labeled with predicate $r$ denotes that $t_1$, $t_2$ are part of the domain and range of $r$, respectively.
This graph allows us to systematically extracts graph patterns, as they were used in the WHERE statement of SPARQL queries.
To generate the query patterns, we consider a subset of 296 common predicates taken from [ReDocRED](https://aclanthology.org/2022.emnlp-main.580/) and [CODRED](https://aclanthology.org/2021.emnlp-main.366/) datasets.
By running the following, we serialize the ontology graph and generate patterns associated to common multi-hop queries:

```shell
python -m scripts.eval.patterns.ontology_graph
python -m scripts.eval.patterns.generate_patterns
```

The generated artificial patterns will be stored in ```data/query-bgps/artificial-patterns.json```.

### Serialize the generated knowledge graphs

To count the patterns in the generated KGs we use [QLever](https://github.com/ad-freiburg/qlever) as an efficient SPARQL endpoint.
For this purpose, we firstly need to serialize the generated KG in .ttl format and index it using QLever, as done in the following:

```shell
python -m scripts.eval.patterns.serialize_graph hotpot1000 \
  --llm openai/gpt-oss-120b --alias {exp alias}
```

You can choose the experiment alias using the flag ```--alias {exp alias}```.
For example, to serialize the KG generated with the OAK method you can use ```--alias oak```.

### Count the number of query patterns and plot the results

After we constructed the query patterns datasets, we can now count the number of query patterns by running:

```shell
python -m scripts.eval.patterns.count_patterns hotpot1000 \
  --alias {exp alias} \
  --llm openai/gpt-oss-120b \
  --patterns-limit 10000 \
  --patterns-filepath {patterns JSON file}
```

Note that the ```--patterns-limit``` is used to randomly sample a subset of BGPs for efficiency reasons.
The counts will be stored in JSON files under ```results/qcounts-{exp alias}```, e.g., ```results/qcounts-oak```.
You can choose the patterns JSON file among the ones retrieved above, e.g., it can be one of:

- ```data/query-bgps/lsq-bgp-patterns.json```: the real-world query patterns obtained from the LSQ-2.0 dataset.
- ```data/query-bgps/artificial-patterns.json```: the artificial query patterns constructed by looking at domain-range constraints of predicates.

Moreover, to reduce the computational cost, note that in the above we can limit the number of patterns per query structure by randomly selecting a subset of $10^4$ patterns for each pattern structure.
Next, we can print the query pattern results (raw counts and our h-index metrics) by running the following:

```shell
python -m scripts.eval.patterns.hscores hotpot1000 \
  --aliases {list of exp aliases, separated by semicolon} \
  --labels {list of labels to use in the legend, separated by semicolon} \
  --llm openai/gpt-oss-120b \
  --md-path "results.md" \
  --json-path "results.json"
```

For example, you can run the following to print the query pattern results of several experiments, each specified by its own alias:

```shell
python -m scripts.eval.patterns.hscores hotpot1000 \
  --aliases "wikontic;inctx;oak;oak-mend-qualifiers" \
  --labels "Wikontic;In-Context;OAK;OAK+MEND" \
  --llm openai/gpt-oss-120b \
  --md-path "results.md" \
  --json-path "results.json"
```

This will print the results in two formats: Markdown (in ```results.md```) and in JSON (in ```results.json```).



# License and Attributions

This project is licensed under the **MIT License**. See the [LICENSE](LICENSE.md) file for more information.

## Original Repository

This repository is a fork of the following project in the [SamsungLabs](https://github.com/SamsungLabs) GitHub organization:

*   **[OKGC](https://github.com/SamsungLabs/OKGC)**: The original repo containing the core implementation and datasets.


## External Code

This repository incorporates or adapts code from the following open-source projects:

*   **[Wikontic](https://github.com/screemix/Wikontic)**, which is licensed under the [MIT License](https://github.com/screemix/Wikontic/blob/main/LICENSE).
*   **[KG-GEN](https://github.com/stair-lab/kg-gen)**, which is licensed under the [MIT License](https://github.com/stair-lab/kg-gen/blob/main/README.md).

## Citation

If you find this work useful for your research, please consider citing our paper:

```bibtex
@article{loconte2026betterlater,
  title={Better Later Than Sooner: Neuro-Symbolic Knowledge Graph Construction via Ontology-grounded Post-extraction Correction},
  author={Loconte, Lorenzo and Hospedales, Timothy and Cornelio, Cristina},
  journal={arXiv preprint arXiv:XXXX.XXXXX},
  year={2026}
}
```
