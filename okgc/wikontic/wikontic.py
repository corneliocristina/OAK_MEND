import logging
import os
from collections import defaultdict

from okgc.utils.sent_embed import SentenceEmbedder

logger = logging.getLogger("StructuredInferenceWithDB")
logger.setLevel(logging.DEBUG)

from pymongo import MongoClient

from okgc.utils.graph import Graph, Qualifier, Triple
from okgc.utils.sparql import (
    Predicate,
    TypeInfo,
)
from okgc.utils.usage import UsageInfo
from okgc.utils.vector_index import PredicatesVectorIndex, TypesVectorIndex
from okgc.wikontic.openai_utils import LLMTripletExtractor
from okgc.wikontic.structured_aligner import Aligner as StructuredDBAligner
from okgc.wikontic.structured_inference_with_db import StructuredInferenceWithDB


def get_mongo_client(mongo_uri):
    client = MongoClient(mongo_uri)
    logger.info("Connection to MongoDB successful")
    return client


class Wikontic:
    def __init__(
        self,
        model: str,
        base_url: str,
        api_key: str,
        types_index: TypesVectorIndex,
        predicates_index: PredicatesVectorIndex,
        sent_embed: SentenceEmbedder,
        prompts_path: str | None = None,
        max_attempts: int = 3,
        seed: int = 42,
        mongodb_uri: str = "mongodb://localhost:27018?directConnection=true",
    ):
        if prompts_path is None:
            prompts_path = os.path.join("prompts", "wikontic")
        self.types_index = types_index
        self.predicates_index = predicates_index

        extractor = LLMTripletExtractor(
            model=model,
            base_url=base_url,
            api_key=api_key,
            prompt_folder_path=prompts_path,
            max_attempts=max_attempts,
            seed=seed,
        )

        ontology_db_name = "wikidata_ontology"
        triplets_db_name = "triplets_db"
        aliases_db_name = "entity_aliases"

        self.mongo_client = get_mongo_client(mongodb_uri)
        self.ontology_db = self.mongo_client.get_database(ontology_db_name)
        self.triplets_db = self.mongo_client.get_database(triplets_db_name)
        self.aliases_collection = self.triplets_db.get_collection(aliases_db_name)

        aligner = StructuredDBAligner(
            ontology_db=self.ontology_db,
            triplets_db=self.triplets_db,
            sent_embed=sent_embed,
        )
        self.engine = StructuredInferenceWithDB(
            extractor, aligner=aligner, triplets_db=self.triplets_db
        )

    def drop_mongo_db_rows(self, sample_id: int):
        for collection in [
            "entity_aliases",
            "initial_triplets",
            "filtered_triplets",
            "ontology_filtered_triplets",
            "triplets",
        ]:
            self.triplets_db.get_collection(collection).delete_many(
                {"sample_id": {"$eq": str(sample_id)}}
            )

    def retrieve_entity_aliases(self, sample_id: int) -> dict[str, list[str]]:
        aliases = defaultdict(set)
        cursors = list(
            self.aliases_collection.find(
                {"sample_id": {"$eq": str(sample_id)}},
                {"_id": False, "label": True, "alias": True},
            )
        )
        for cursor in cursors:
            aliases[cursor["label"]].add(cursor["alias"])
        return {l: list(s) for l, s in aliases.items()}

    def generate_from_text(
        self, text: str, *, text_id: int = 0
    ) -> tuple[Graph, dict[str, UsageInfo]]:
        sample_id = text_id >> 16
        source_text_id = sample_id & 0xFFFF
        _, final_triplets, _, ontology_filtered_triplets, usage_info = (
            self.engine.extract_triplets_with_ontology_filtering(
                text, sample_id=str(sample_id), source_text_id=str(source_text_id)
            )
        )
        extracted_triples = final_triplets + ontology_filtered_triplets

        # Make sure all entities are strings
        for t in extracted_triples:
            for k in ["subject", "object"]:
                if not isinstance(t[k], str):
                    t[k] = str(t[k])
            for q in t["qualifiers"]:
                if not isinstance(q["object"], str):
                    q["object"] = str(q["object"])

        # Collect the triples and the qualifiers
        triples: set[Triple] = set()
        predicates: dict[str, Predicate] = {}
        qualifiers: dict[Triple, list[Qualifier]] = defaultdict(list)
        for t in extracted_triples:
            p_name = str(t["relation"])
            if t["relation_id"] is None:
                p = Predicate.unknown_from_name(p_name)
                p_id = p.code
            else:
                p_id = t["relation_id"]
                p = self.predicates_index[p_id]
            predicates[p_id] = p
            new_t = (t["subject"], p_id, t["object"])
            triples.add(new_t)
            # Qualifier predicates are not mapped to the ontology
            # So, tag them as unknown predicates
            qs = []
            for qd in t["qualifiers"]:
                q_name = str(qd["relation"])
                q = Predicate.unknown_from_name(q_name)
                qs.append((q.code, qd["object"]))
                predicates[q.code] = q
                qualifiers[new_t].extend(qs)

        # Collect all the entities
        entities: set[str] = set(t[0] for t in triples) | set(t[2] for t in triples)
        # entities |= set(q[1] for qs in qualifiers.values() for q in qs)

        # Collect all the entity types
        entity_types: dict[str, list[TypeInfo]] = defaultdict(list)
        for t in extracted_triples:
            s = t["subject"]
            o = t["object"]
            s_type_name = str(t["subject_type"])
            o_type_name = str(t["object_type"])
            if (
                t["subject_type_id"] is None
                or t["subject_type_id"] not in self.types_index
            ):
                s_type = TypeInfo.unknown_from_name(s_type_name)
            else:
                s_type = self.types_index[t["subject_type_id"]]
            if (
                t["object_type_id"] is None
                or t["object_type_id"] not in self.types_index
            ):
                o_type = TypeInfo.unknown_from_name(o_type_name)
            else:
                o_type = self.types_index[t["object_type_id"]]
            if s_type.code not in [u.code for u in entity_types[s]]:
                entity_types[s].append(s_type)
            if o_type.code not in [u.code for u in entity_types[o]]:
                entity_types[o].append(o_type)

        # Construct the pointers to the text ids
        triple_text_ids: dict[Triple, list[int]] = {t: [text_id] for t in triples}
        entity_text_ids: dict[str, list[int]] = {e: [text_id] for e in entities}

        graph = Graph(
            entities,
            entity_types,
            entity_descriptions=None,
            predicates=predicates,
            triples=triples,
            entity_text_ids=entity_text_ids,
            triple_text_ids=triple_text_ids,
            qualifiers=qualifiers,
        )

        return graph, usage_info
