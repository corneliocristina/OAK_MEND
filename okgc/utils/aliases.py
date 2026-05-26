from collections import defaultdict

import spacy

from okgc.utils.graph import Graph
from okgc.utils.strings import simplify_entity_alias


def spacy_find_entity_aliases(
    kg: Graph, *, nlp: spacy.Language
) -> dict[str, list[str]]:
    # Cluster the entities based on a simplified alias
    # obtained via a NLP pipeline (lemmatization + removal of stopwords + removal of punctuation)
    alias_entity_clusters: dict[str, set[str]] = defaultdict(set)
    for e in kg.entities:
        e_alias = simplify_entity_alias(e, nlp=nlp)
        alias_entity_clusters[e_alias].add(e)
    # Select the representative entity in each cluster,
    # by simply taking the entity having the longest alias
    alias_entity_cluster_representatives: dict[str, str] = {
        e: sorted(cluster, key=lambda x: len(x), reverse=True)[0]
        for e, cluster in alias_entity_clusters.items()
    }
    # Cluster the entities in a dictionary with the representative entities as the keys
    return {
        alias_entity_cluster_representatives[e]: list(cluster)
        for e, cluster in alias_entity_clusters.items()
    }


def spacy_nlp_tokenize(corpus: str, *, nlp: spacy.Language) -> list[str]:
    return [doc.lemma_ for doc in nlp(corpus)]
