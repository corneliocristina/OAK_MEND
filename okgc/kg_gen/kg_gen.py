###############################################################################
#
# okgc/kg_gen/kg_gen.py
#
# Code originally from src/kg_gen/kg_gen.py in
# https://github.com/stair-lab/kg-gen/tree/main/src/kg_gen
# 7b8c814575f143c3c0a8e0b26b832b7332445dde
#
# MIT License
#
###############################################################################

import json

# Configure dspy logging to only show errors
import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Union

import dspy
import networkx as nx
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from typing_extensions import deprecated

from okgc.kg_gen.models import Graph
from okgc.kg_gen.steps._1_get_entities import get_entities
from okgc.kg_gen.steps._2_get_relations import get_relations
from okgc.kg_gen.steps._3_deduplicate import DeduplicateMethod, run_deduplication
from okgc.kg_gen.utils.chunk_text import chunk_text

logger = logging.getLogger(__name__)

dspy_logger = logging.getLogger("dspy")
dspy_logger.setLevel(logging.CRITICAL)


class KGGen:
    def __init__(
        self,
        model: str = "openai/gpt-4o",
        retrieval_model: SentenceTransformer | None = None,
        temperature: float = None,
        reasoning_effort: str = None,
        api_key: str = None,
        api_base: str = None,
        disable_cache: bool = False,
        seed: int = 42,
    ):
        """Initialize KGGen with optional model configuration

        Args:
            model: Name of model to use (e.g. 'gpt-4')
            temperature: Temperature for model sampling
            api_key: API key for model access
            api_base: Specify the base URL endpoint for making API calls to a language model service
        """
        assert retrieval_model is not None
        self.model = model
        self.reasoning_effort = reasoning_effort
        self.temperature = temperature
        self.api_key = api_key
        self.api_base = api_base
        self.retrieval_model = retrieval_model
        self.lm = None
        self.disable_cache = disable_cache
        self.seed = seed

        self.init_model(
            model=model,
            reasoning_effort=reasoning_effort,
            temperature=temperature,
            api_key=api_key,
            api_base=api_base,
        )

    def validate_temperature(self, temperature: float):
        if "gpt-5" in self.model and temperature < 1.0:
            raise ValueError("Temperature must be 1.0 for gpt-5 family models")

    def init_model(
        self,
        model: str = None,
        reasoning_effort: str = None,
        temperature: float = None,
        api_key: str = None,
        api_base: str = None,
    ):
        """Initialize or reinitialize the model with new parameters

        Args:
            model: Name of model to use (e.g. 'gpt-4')
            temperature: Temperature for model sampling
            api_key: API key for model access
            api_base: API base for model access
            reasoning_effort: Reasoning effort for model
            temperature: Temperature for model sampling
        """

        # Update instance variables if new values provided
        if model is not None:
            self.model = model
        if api_key is not None:
            self.api_key = api_key
        if api_base is not None:
            self.api_base = api_base
        if temperature is not None:
            self.temperature = temperature
        if reasoning_effort is not None:
            self.reasoning_effort = reasoning_effort

        self.validate_temperature(self.temperature)

        # Initialize dspy LM with current settings
        if self.api_key:
            self.lm = dspy.LM(
                model=self.model,
                api_key=self.api_key,
                reasoning={"effort": self.reasoning_effort}
                if self.reasoning_effort
                else None,
                temperature=self.temperature,
                api_base=self.api_base,
                cache=not self.disable_cache,
                model_type="responses" if self.model.startswith("openai/") else "chat",
                seed=self.seed,
            )
        else:
            self.lm = dspy.LM(
                model=self.model,
                temperature=self.temperature,
                api_base=self.api_base,
                reasoning={"effort": self.reasoning_effort}
                if self.reasoning_effort
                else None,
                cache=not self.disable_cache,
                model_type="responses" if self.model.startswith("openai/") else "chat",
                seed=self.seed,
            )

    @staticmethod
    def from_file(file_path: str) -> Graph:
        with open(file_path, "r") as f:
            graph = Graph(**json.load(f))
        return graph

    @staticmethod
    def from_dict(graph_dict: dict) -> Graph:
        return Graph(**graph_dict)

    def generate(
        self,
        input_data: Union[str, List[Dict]],
        model: str = None,
        api_key: str = None,
        api_base: str = None,
        context: str = "",
        chunk_size: Optional[int] = None,
        reasoning_effort: str = None,
        deduplication_method: DeduplicateMethod | None = DeduplicateMethod.SEMHASH,
        temperature: float = None,
        output_folder: Optional[str] = None,
        no_dspy: bool = False,
    ) -> Graph:
        """Generate a knowledge graph from input text or messages.

        Args:
            input_data: Text string or list of message dicts
            model: Name of OpenAI model to use
            api_key (str): OpenAI API key for making model calls
            chunk_size: Max size of text chunks in characters to process
            context: Description of data context
            output_folder: Path to save partial progress

        Returns:
            Graph: Generated knowledge graph
        """

        # Process input data
        is_conversation = isinstance(input_data, list)
        if is_conversation:
            # Extract text from messages
            text_content = []
            for message in input_data:
                if (
                    not isinstance(message, dict)
                    or "role" not in message
                    or "content" not in message
                ):
                    raise ValueError(
                        "Messages must be dicts with 'role' and 'content' keys"
                    )
                if message["role"] in ["user", "assistant"]:
                    text_content.append(f"{message['role']}: {message['content']}")

            # Join with newlines to preserve message boundaries
            processed_input = "\n".join(text_content)
        else:
            processed_input = input_data

        # Reinitialize dspy with new parameters if any are provided
        if any([model, temperature, api_key, api_base, reasoning_effort]):
            self.init_model(
                model=model or self.model,
                temperature=temperature or self.temperature,
                api_key=api_key or self.api_key,
                api_base=api_base or self.api_base,
                reasoning_effort=reasoning_effort or self.reasoning_effort,
            )

        def _process(content, lm):
            with dspy.context(lm=lm, track_usage=True):
                entities = get_entities(
                    content,
                    is_conversation,
                    use_litellm_prompt=no_dspy,
                    model=self.model,
                    api_key=self.api_key,
                    api_base=self.api_base,
                    temperature=temperature
                    if temperature is not None
                    else self.temperature,
                )
                relations = get_relations(
                    content,
                    entities,
                    is_conversation=is_conversation,
                    use_litellm_prompt=no_dspy,
                    model=self.model,
                    api_key=self.api_key,
                    api_base=self.api_base,
                    temperature=temperature
                    if temperature is not None
                    else self.temperature,
                )
                return entities, relations

        if not chunk_size:
            try:
                entities, relations = _process(processed_input, self.lm)
            except Exception as e:
                if "context length" in str(e).lower():
                    logger.warning(
                        f"Context length error: {e}. Chunking text with chunk size 16384."
                    )
                    chunk_size = 16384
                else:
                    raise e

        if chunk_size:
            chunks = chunk_text(processed_input, chunk_size)
            entities = set()
            relations = set()

            with ThreadPoolExecutor() as executor:
                future_to_chunk = {
                    executor.submit(_process, chunk, self.lm): chunk for chunk in chunks
                }

                for future in as_completed(future_to_chunk):
                    chunk_entities, chunk_relations = future.result()
                    entities.update(chunk_entities)
                    relations.update(chunk_relations)

        graph = Graph(
            entities=entities,
            relations=relations,
            edges={relation[1] for relation in relations},
        )

        if deduplication_method:
            graph = self.deduplicate(
                graph, method=deduplication_method, context=context
            )

        if output_folder:
            self.export_graph(graph, os.path.join(output_folder, "graph.json"))
        return graph

    @deprecated("Use KGGen.deduplicate() method instead")
    def cluster(
        self,
        graph: Graph,
        **kwargs,
    ) -> Graph:
        return self.deduplicate(graph, **kwargs)

    def deduplicate(
        self,
        graph: Graph,
        method: DeduplicateMethod = DeduplicateMethod.FULL,
        semhash_similarity_threshold: float = 0.95,  # recommended to keep at 0.95
        model: str = None,
        temperature: float = None,
        api_key: str = None,
        api_base: str = None,
        context: str = "",  # TODO: implement context
    ) -> Graph:
        # Reinitialize dspy with new parameters if any are provided
        if any([model, temperature, api_key, api_base]):
            self.init_model(
                model=model or self.model,
                temperature=temperature or self.temperature,
                api_key=api_key or self.api_key,
                api_base=api_base or self.api_base,
            )

        return run_deduplication(
            lm=self.lm,
            graph=graph,
            method=method,
            retrieval_model=self.retrieval_model,
            semhash_similarity_threshold=semhash_similarity_threshold,
        )

    def aggregate(self, graphs: list[Graph]) -> Graph:
        # Initialize empty sets for combined graph
        all_entities = set()
        all_relations = set()
        all_edges = set()
        all_entity_metadata: dict[str, set[str]] = {}

        # Combine all graphs
        for graph in graphs:
            all_entities.update(graph.entities)
            all_relations.update(graph.relations)
            all_edges.update(graph.edges)
            if graph.entity_metadata:
                for entity, metadata_set in graph.entity_metadata.items():
                    if entity in all_entity_metadata:
                        all_entity_metadata[entity].update(metadata_set)
                    else:
                        all_entity_metadata[entity] = metadata_set.copy()

        # Create and return aggregated graph
        return Graph(
            entities=all_entities,
            relations=all_relations,
            edges=all_edges,
            entity_metadata=all_entity_metadata if all_entity_metadata else None,
        )

    # @staticmethod
    # def visualize(graph: Graph, output_path: str, open_in_browser: bool = False):
    #     visualize_kg(graph, output_path, open_in_browser=open_in_browser)

    # ====== Retrieval Methods ======

    def _parse_embedding_model(
        self, model: Optional[SentenceTransformer] = None
    ) -> Optional[SentenceTransformer]:
        if model is None:
            model = self.retrieval_model
        if model is None:
            raise ValueError("No retrieval model provided")
        return model

    @staticmethod
    def to_nx(graph: Graph) -> nx.DiGraph:
        G = nx.DiGraph()
        for entity in graph.entities:
            G.add_node(entity)

        for relation in graph.relations:
            source, rel, target = relation
            G.add_edge(source, target, relation=rel)
        return G

    def generate_embeddings(
        self,
        graph: Union[Graph, nx.DiGraph],
        model: Optional[SentenceTransformer] = None,
    ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        model = self._parse_embedding_model(model)
        if isinstance(graph, Graph):
            graph = self.to_nx(graph)

        node_embeddings = {node: model.encode(node).tolist() for node in graph.nodes}
        relation_embeddings = {
            rel: model.encode(rel).tolist()
            # TODO: this is triggering index out of range error
            for rel in set(edge[2]["relation"] for edge in graph.edges(data=True))
        }
        return node_embeddings, relation_embeddings

    def retrieve(
        self,
        query: str,
        node_embeddings: dict[str, np.ndarray],
        graph: nx.DiGraph,
        model: Optional[SentenceTransformer] = None,
        k: int = 8,
        verbose: bool = False,
    ) -> tuple[list[tuple[str, float]], set[str], str]:
        model = self._parse_embedding_model(model)
        top_nodes = self.retrieve_relevant_nodes(query, node_embeddings, model, k)
        context = set()
        for node, _ in top_nodes:
            node_context = self.retrieve_context(node, graph)
            if verbose:
                print(f"Context for node {node}: {node_context}")
            context.update(node_context)
        context_text = " ".join(context)
        if verbose:
            print(f"Combined context: '{context_text}'\n---")
        return top_nodes, context, context_text

    @staticmethod
    def retrieve_relevant_nodes(
        query: str,
        node_embeddings: dict[str, np.ndarray],
        model: SentenceTransformer,
        k: int = 8,
    ) -> list[tuple[str, float]]:
        query_embedding = model.encode(query).reshape(1, -1)
        similarities = []
        for node, embed in node_embeddings.items():
            target_embedding = np.array(embed).reshape(1, -1)
            similarity = cosine_similarity(query_embedding, target_embedding)[0][0]
            similarities.append((node, similarity))
        similarities = sorted(similarities, key=lambda x: x[1], reverse=True)
        return similarities[:k]

    @staticmethod
    def retrieve_context(node: str, graph: nx.DiGraph, depth: int = 2) -> list[str]:
        context = set()

        def explore_neighbors(current_node, current_depth):
            if current_depth > depth:
                return
            # Outgoing edges
            for neighbor in graph.neighbors(current_node):
                rel = graph[current_node][neighbor]["relation"]
                context.add(f"{current_node} {rel} {neighbor}.")
                explore_neighbors(neighbor, current_depth + 1)
            # Incoming edges
            for neighbor in graph.predecessors(current_node):
                rel = graph[neighbor][current_node]["relation"]
                context.add(f"{neighbor} {rel} {current_node}.")
                explore_neighbors(neighbor, current_depth + 1)

        explore_neighbors(node, 1)
        return list(context)

    @staticmethod
    def graph_to_dict(graph: Graph) -> dict:
        return {
            "entities": list(graph.entities),
            "relations": list(graph.relations),
            "edges": list(graph.edges),
            "entity_clusters": {k: list(v) for k, v in graph.entity_clusters.items()}
            if graph.entity_clusters
            else None,
            "edge_clusters": {k: list(v) for k, v in graph.edge_clusters.items()}
            if graph.edge_clusters
            else None,
            "entity_metadata": graph.entity_metadata,
        }

    @staticmethod
    def dict_to_graph(data: dict) -> Graph:
        return Graph(
            entities=set(data["entities"]),
            relations=set(data["relations"]),
            edges=set(data["edges"]),
            entity_clusters=data["entity_clusters"],
            edge_clusters=data["edge_clusters"],
            entity_metadata=data["entity_metadata"],
        )

    @staticmethod
    def export_graph(graph: Graph, output_path: str):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        graph_dict = KGGen.graph_to_dict(graph)

        with open(output_path, "w") as f:
            json.dump(graph_dict, f, indent=2)

    # ====== Token Usage ======
    def reset_token_usage(self):
        self.lm.history = []

    def extract_token_usage_from_history(self) -> Dict[str, int]:
        """Extract token usage from dspy LM history."""

        total_prompt_tokens = 0
        total_completion_tokens = 0
        total_tokens = 0

        for entry in self.lm.history:
            if isinstance(entry, dict):
                # Check for usage information in various possible locations
                usage = entry.get("usage") or entry.get("response", {}).get("usage")

                if usage:
                    total_prompt_tokens += usage.get("prompt_tokens", 0)
                    total_completion_tokens += usage.get("completion_tokens", 0)
                    total_tokens += usage.get("total_tokens", 0)

        return {
            "prompt_tokens": total_prompt_tokens,
            "completion_tokens": total_completion_tokens,
            "total_tokens": total_tokens,
        }
