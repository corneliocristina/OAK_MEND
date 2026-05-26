import json
import os
import re
import traceback
from collections import defaultdict
from dataclasses import asdict

import networkx as nx
import rdflib
from tqdm import tqdm

from okgc.datasets.loaders import load_predicates
from okgc.utils.query_patterns import BGP, extract_triples_from_bgp_body
from okgc.utils.sparql import WIKIDATA_PROPERTY_DIRECT_PREFIX, sparql_query

ENDPOINT = "http://lsq.aksw.org/sparql"

LIMIT = 5000

MAX_QUERY_ITERATIONS = 250

QUERY = """
PREFIX lsqv: <http://lsq.aksw.org/vocab#>

SELECT DISTINCT ?bgp ?bgpContent WHERE {
    ?query lsqv:hasRemoteExec/lsqv:endpoint  <https://query.wikidata.org/> ;
           lsqv:hasStructuralFeatures ?features .

    ?features lsqv:hasBgp ?bgp ;
              lsqv:joinVertexCount ?joinVertexCount .
              
    ?bgp rdfs:label ?bgpContent .

    FILTER(?joinVertexCount > 0)
}
"""

MAX_DOUBLE_QMARK_VARS = 50


def add_datatype_to_floats(body: str) -> str:
    """
    Fix negative decimal literals in SPARQL queries by converting them to typed literals.
    """
    # Pattern explanation:
    # (?<![:\w./-]) - negative lookbehind: not preceded by :, word chars, ., /, -, or "
    # -              - literal minus sign
    # (\d+\.\d+)    - capture group: digits.digits
    # (?![>\w./-])  - negative lookahead: not followed by >, word chars, ., /, -, or "

    pattern = r'(?<![:\w./\-"])-(\d+\.\d+)(?![>\w./\-"])'
    # Replace with typed literal
    replacement = r'"\1"^^<http://www.w3.org/2001/XMLSchema#double>'
    result = re.sub(pattern, replacement, body)
    return result


def normalize_bgp_pattern(body: str) -> tuple[nx.DiGraph, str, int, int] | None:
    # Clean and parse the BGP pattern string
    body = re.sub(r"\s+", " ", body.strip())
    processed_body = str(body)
    if "??" in processed_body:
        var_idx = 1
        for i in range(MAX_DOUBLE_QMARK_VARS):
            double_q_var = f"??{i}"
            if double_q_var not in processed_body:
                continue
            processed_body = processed_body.replace(double_q_var, f"?x{var_idx}")
            var_idx += 1
    processed_body = add_datatype_to_floats(processed_body)
    try:
        triples = extract_triples_from_bgp_body(processed_body)
    except:
        print(
            f"BGP normalization failed for:\n{processed_body}\nOriginal BGP is:\n{body}\nThe exception is:\n{traceback.format_exc()}"
        )
        return None
    entities = set(t[0] for t in triples) | set(t[2] for t in triples)
    predicates = set(t[1] for t in triples)

    # Consider only BGP with more than one distinct predicate
    if len(predicates) <= 1:
        return None
    # Also, for some BGPs after we replace subject/object terms with fresh variables
    # we obtain many triples that are equivalent, e.g., the triple pattern
    #    ?x1 wdt:Pxxx ?x2 . ?x1 wdt:Pxxx ?x3 . ?x2 wdt:Pxxx ?x4 .
    # These BGPs without constant subject/object terms are less interesting
    # and would make the counts explode combinatorially
    # Thus, here we consider only BGPs whose triples have distinct predicates
    if len(predicates) != len(triples):
        return None

    # Assign indices to the entities/variables
    entities_idx: dict[str, int] = {}
    cur_entity_idx = 0
    for s, r, o in triples:
        e_s = str(s)
        if e_s not in entities_idx:
            entities_idx[e_s] = cur_entity_idx
            cur_entity_idx += 1
        e_o = str(o)
        if e_o not in entities_idx:
            entities_idx[e_o] = cur_entity_idx
            cur_entity_idx += 1

    # Replace all predicates with "wdt:Pxxx" shorthands
    predicates_map = {}
    for r in predicates:
        # Skip if some triples predicates are actually paths
        if isinstance(r, rdflib.paths.Path):
            return None
        # Skip if a predicate is not on wikidata.org
        if f"{WIKIDATA_PROPERTY_DIRECT_PREFIX}P" not in r:
            return None
        p = str(r).split("/")[-1]
        # Skip if there is some evident typo
        if p[0] != "P" or len(p) <= 1:
            return None
        # Skip if there is a predicate not in the list of allowed ones we use in the experiments
        if p not in allowed_predicates:
            return None
        predicates_map[r] = p

    # Normalize the triples and produce a normalized BGP body
    normalized_triples = []
    for s, r, o in triples:
        p = predicates_map[r]
        v1 = f"x{entities_idx[str(s)]}"
        v2 = f"x{entities_idx[str(o)]}"
        normalized_triples.append((v1, p, v2))
    normalized_body = " ".join(
        f"?{v1} wdt:{p} ?{v2} ." for v1, p, v2 in normalized_triples
    )

    # Construct the NetworkX graph
    g = nx.DiGraph()
    for s, r, o in triples:
        g.add_edge(s, o, predicate=r)

    # Compute BGP stats
    num_triples = len(triples)
    entity_arities: dict[str, int] = defaultdict(int)
    for t in triples:
        s, r, o = t
        entity_arities[str(s)] += 1
        entity_arities[str(o)] += 1
    max_arity = max(entity_arities.values())
    return g, normalized_body, num_triples, max_arity


if __name__ == "__main__":
    # Load the list of allowed predicates
    predicates, _ = load_predicates("hotpot1000", datapath="data")
    allowed_predicates = set(predicates.keys())

    # Load or retrieve the raw LSQ basic graph pattern (BGP) queries
    lsq_root_path = os.path.join("data", "lsq")
    os.makedirs(lsq_root_path, exist_ok=True)
    lsq_raw_dump_filepath = os.path.join(lsq_root_path, "lsq-raw-dump.json")
    if not os.path.isfile(lsq_raw_dump_filepath):
        offset = 0
        continue_querying = True
        bindings = []
        it = 0
        while continue_querying:
            query = f"{QUERY}\nLIMIT {LIMIT}\n OFFSET {offset}"
            try:
                output = sparql_query(endpoint=ENDPOINT, query=query)
                bindings.extend(output["results"]["bindings"])
                it += 1
                offset += LIMIT
                continue_querying = (
                    len(output["results"]["bindings"]) == LIMIT
                    and it < MAX_QUERY_ITERATIONS
                )
                print(f"Iteration #{it}: retrieved {len(bindings)} BGP rows")
            except:
                print(f"Exiting because of exception:\n{traceback.format_exc()}")
                continue_querying = False
        lsq: dict[str, str] = {}
        for binding in bindings:
            bgp_id = binding["bgp"]["value"].split("/")[-1][9:]
            body = binding["bgpContent"]["value"]
            if bgp_id in lsq:
                continue
            lsq[bgp_id] = body
        with open(lsq_raw_dump_filepath, "w") as fp:
            json.dump(lsq, fp)
    with open(lsq_raw_dump_filepath, "r") as fp:
        lsq = json.load(fp)
    print(f"Number of raw BGP queries: {len(lsq)}")

    # Normalize and filter the BGPs
    seen_digraphs: set[nx.DiGraph] = set()
    normalized_bgps: set[BGP] = set()
    tqdm_lsq = tqdm(lsq.items())
    for bgp_id, bgp_body in tqdm_lsq:
        normalize_output = normalize_bgp_pattern(bgp_body)
        if normalize_output is None:
            continue
        g, normalized_bgp_body, num_triples, max_arity = normalize_output
        # We skip BGPs consisting of a single connected component
        if nx.number_weakly_connected_components(g) != 1:
            continue
        # Check for duplicate BGPs via isomorphism check
        if any(
            nx.is_isomorphic(
                g,
                h,
                edge_match=lambda e1, e2: e1.get("predicate") == e2.get("predicate"),
            )
            for h in seen_digraphs
        ):
            continue
        bgp = BGP(normalized_bgp_body, num_triples, max_arity)
        normalized_bgps.add(bgp)
        seen_digraphs.add(g)
        tqdm_lsq.set_description(f"Valid BGPs: {len(normalized_bgps)}")
    assert len(normalized_bgps) == len(seen_digraphs)
    print(f"Number of normalized BGP queries: {len(normalized_bgps)}")

    # Save the processed BGPs to file
    query_patterns_path = os.path.join("data", "query-bgps")
    os.makedirs(query_patterns_path, exist_ok=True)
    lsq_bgp_patterns_filepath = os.path.join(
        query_patterns_path, "lsq-bgp-patterns.json"
    )
    with open(lsq_bgp_patterns_filepath, "w") as fp:
        formatted_bgps = {"lsq": [asdict(bgp) for bgp in normalized_bgps]}
        json.dump(formatted_bgps, fp)
