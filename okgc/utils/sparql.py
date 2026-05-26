import re
import sys
from dataclasses import dataclass
from enum import IntEnum, auto

from SPARQLWrapper import JSON, SPARQLWrapper
from tenacity import retry, stop_after_attempt, wait_random_exponential


@dataclass(frozen=True)
class TypeInfo:
    code: str
    name: str
    description: str | None = None
    aliases: list[str] | None = None

    def basename(self) -> str:
        # Some different types in Wikidata have the same name
        # Currently, we avoid types having different names by appending a description in parentheses
        # This method returns the type name _without_ the appended description
        return re.sub(r"\(.+\)", "", self.name, flags=re.DOTALL).strip()

    def __repr__(self) -> str:
        if self.is_unknown():
            return f"{self.code}"
        return f"{self.basename()} ({self.code})"

    def is_unknown(self) -> bool:
        return self.code[:4] == "QUNK"

    @staticmethod
    def unknown_from_name(name: str, description: str | None = None) -> "TypeInfo":
        code = f"QUNK ({name})"
        return TypeInfo(code, name, description=description, aliases=None)


class PredicateKind(IntEnum):
    UNKNOWN = auto()
    ITEM = auto()
    POINT_IN_TIME = auto()
    QUANTITY = auto()


@dataclass(frozen=True)
class Predicate:
    code: str
    name: str
    kind: PredicateKind
    domain: list[str]
    range: list[str]
    description: str | None = None
    aliases: list[str] | None = None
    qualifiers: list[str] | None = None
    candidate_code: str | None = None

    def __repr__(self) -> str:
        if self.is_unknown():
            return f"{self.code}"
        return f"{self.name} ({self.code})"

    def is_unknown(self) -> bool:
        return self.code[:4] == "PUNK"

    @staticmethod
    def unknown_from_name(
        name: str, *, candidate_code: str | None = None, description: str | None = None
    ) -> "Predicate":
        code = (
            f"PUNK ({name})"
            if candidate_code is None
            else f"PUNK ({candidate_code}) ({name})"
        )
        return Predicate(
            code,
            name,
            PredicateKind.UNKNOWN,
            [],
            [],
            description=description,
            aliases=None,
            qualifiers=None,
            candidate_code=candidate_code,
        )


SPECIAL_TYPE_CODES: dict[str, str] = {
    "point in time": "Q186408",
    "quantity": "Q309314",
    "number": "Q11563",
}


class TypesHierarchy:
    def __init__(self, types: dict[str, TypeInfo], superclasses: dict[str, set[str]]):
        assert types.keys() == superclasses.keys()
        self.types = types
        self.superclasses = superclasses

    def __contains__(self, code: str) -> bool:
        return code in self.types

    def __getitem__(self, code: str) -> TypeInfo:
        return self.types[code]

    def is_subclass(self, t1: TypeInfo, t2: TypeInfo) -> bool:
        if t1.is_unknown() or t2.is_unknown():
            return False
        if t1.code == t2.code:
            return True
        return t2.code in self.superclasses[t1.code]

    def is_literal(self, t: TypeInfo) -> bool:
        return self.is_point_in_time(t) or self.is_quantity(t) or self.is_number(t)

    def is_point_in_time(self, t: TypeInfo) -> bool:
        if t.is_unknown():
            return False
        return self.is_subclass(t, self.types[SPECIAL_TYPE_CODES["point in time"]])

    def is_quantity(self, t: TypeInfo) -> bool:
        if t.is_unknown():
            return False
        return self.is_subclass(t, self.types[SPECIAL_TYPE_CODES["quantity"]])

    def is_number(self, t: TypeInfo) -> bool:
        if t.is_unknown():
            return False
        return self.is_subclass(t, self.types[SPECIAL_TYPE_CODES["number"]])


SPARQL_QUERIES: dict[str, str] = {
    # Given the id of a property, return its label, its description, and its aliases
    "property_label_description_aliases": """
select ?predicateLabel ?predicateKind ?predicateDescription ?predicateAliases {{
  wd:{code} wikibase:propertyType ?propertyType .
  bind(
    if(?propertyType = wikibase:WikibaseItem, "Item",
      if(?propertyType = wikibase:Quantity, "Quantity",
        if(?propertyType = wikibase:Time, "PointInTime", "Unknown")
      )
    ) AS ?predicateKind
  )
  SERVICE wikibase:label {{
    bd:serviceParam wikibase:language "en" .
    wd:{code} rdfs:label ?predicateLabel ;
              schema:description ?predicateDescription ;
              skos:altLabel ?predicateAliases .
  }}
}}
""",
    # Retrieve the label, description and aliases of an item, given its code
    "item_label_description_aliases": """
select ?itemLabel ?itemDescription ?itemAliases  {{
  SERVICE wikibase:label {{
    bd:serviceParam wikibase:language "en" .
    wd:{code} rdfs:label ?itemLabel ;
              schema:description ?itemDescription ;
              skos:altLabel ?itemAliases .
  }}
}}
""",
    # Retrieve the subject-type constraints of a property given its code
    "property_subject_types": """
select ?subjectType {{
  wd:{code} p:P2302 ?cst .
  ?cst ps:P2302 wd:Q21503250 .  # subject type constraint
  ?cst pq:P2308 ?subjectType
}}
""",
    # Retrieve the value-type constraints of a property given its code
    "property_value_types": """
select ?valueType {{
  wd:{code} p:P2302 ?cvt .
  ?cvt ps:P2302 wd:Q21510865 .  # value-type constraint
  ?cvt pq:P2308 ?valueType
}}
""",
    # Retrieve the qualifier constraints of a property given its code
    "property_qualifiers": """
select ?qualifier {{
  wd:{code} p:P2302 ?cvt .
  ?cvt ps:P2302 wd:Q21510851 .  # allowed-qualifiers constraint
  ?cvt pq:P2306 ?qualifier
}}
""",
    # Retrieve the superclasses of an entity given its code
    "item_superclasses": """
select ?superclass {{
  wd:{code} wdt:P279+ ?superclass .
}}
""",
    # Retrieve the immediate subclasses of an entity
    "item_subclasses": """
select ?subclass {{
    ?subclass wdt:P279 wd:{code} .
}}
""",
    # Check whether there is a path between to types in the subclass hierarchy
    "subclass_path": """
    ask where {{ wd:{codeA} wdt:P279* wd:{codeB} }}
""",
}


def lookup_type_by_code(type_code: str) -> TypeInfo | None:
    query = SPARQL_QUERIES["item_label_description_aliases"].format(code=type_code)
    outputs = sparql_query(query=query)
    bindings = outputs["results"]["bindings"]
    if len(bindings) == 0:
        return None
    assert len(bindings) == 1
    (binding,) = bindings
    name: str = binding["itemLabel"]["value"]
    description: str | None = None
    if "itemDescription" in binding and binding["itemDescription"]["value"]:
        description = binding["itemDescription"]["value"]
    aliases: list[str] | None = None
    if "itemAliases" in binding and binding["itemAliases"]["value"]:
        aliases = list(map(str.strip, binding["itemAliases"]["value"].split(",")))
    return TypeInfo(type_code, name, description, aliases)


def lookup_property_by_code(predicate_code: str) -> Predicate | None:
    query = SPARQL_QUERIES["property_label_description_aliases"].format(
        code=predicate_code
    )
    outputs = sparql_query(query=query)
    bindings = outputs["results"]["bindings"]
    if len(bindings) == 0:
        return None
    assert len(bindings) == 1
    (binding,) = bindings
    predicate_name: str = binding["predicateLabel"]["value"]
    predicate_kind_name: str = binding["predicateKind"]["value"]
    if predicate_kind_name == "Item":
        predicate_kind = PredicateKind.ITEM
    elif predicate_kind_name == "PointInTime":
        predicate_kind = PredicateKind.POINT_IN_TIME
    elif predicate_kind_name == "Quantity":
        predicate_kind = PredicateKind.QUANTITY
    else:
        predicate_kind = PredicateKind.UNKNOWN
    predicate_description: str | None = None
    if "predicateDescription" in binding and binding["predicateDescription"]["value"]:
        predicate_description = binding["predicateDescription"]["value"]
    predicate_aliases: list[str] | None = None
    if "predicateAliases" in binding and binding["predicateAliases"]["value"]:
        predicate_aliases = list(
            map(str.strip, binding["predicateAliases"]["value"].split(","))
        )

    # Retrieve the domain constraints
    query = SPARQL_QUERIES["property_subject_types"].format(code=predicate_code)
    outputs = sparql_query(query=query)
    bindings = outputs["results"]["bindings"]
    domain: list[str] = []
    for binding in bindings:
        type_code = binding["subjectType"]["value"].split("/")[-1]
        domain.append(type_code)

    # Retrieve the range constraints
    query = SPARQL_QUERIES["property_value_types"].format(code=predicate_code)
    outputs = sparql_query(query=query)
    bindings = outputs["results"]["bindings"]
    range: list[str] = []
    for binding in bindings:
        type_code = binding["valueType"]["value"].split("/")[-1]
        range.append(type_code)
    # Deal with range being composed by point in time or by a quantity
    if not range:
        if predicate_kind == PredicateKind.POINT_IN_TIME:
            range.append(SPECIAL_TYPE_CODES["point in time"])
        elif predicate_kind == PredicateKind.QUANTITY:
            range.append(SPECIAL_TYPE_CODES["quantity"])

    # Retrieve the qualifiers
    query = SPARQL_QUERIES["property_qualifiers"].format(code=predicate_code)
    outputs = sparql_query(query=query)
    bindings = outputs["results"]["bindings"]
    qualifiers: list[str] | None = None
    if bindings:
        qualifiers = []
        for binding in bindings:
            qualifier_code = binding["qualifier"]["value"].split("/")[-1]
            qualifiers.append(qualifier_code)

    return Predicate(
        predicate_code,
        predicate_name,
        predicate_kind,
        domain,
        range,
        description=predicate_description,
        aliases=predicate_aliases,
        qualifiers=qualifiers,
    )


def collect_superclasses(t: TypeInfo) -> list[str]:
    query = SPARQL_QUERIES["item_superclasses"].format(code=t.code)
    outputs = sparql_query(query=query)
    bindings = outputs["results"]["bindings"]
    superclass_codes = []
    for binding in bindings:
        code = binding["superclass"]["value"].split("/")[-1]
        superclass_codes.append(code)
    return superclass_codes


def collect_subclasses(t: TypeInfo) -> list[str]:
    query = SPARQL_QUERIES["item_subclasses"].format(code=t.code)
    outputs = sparql_query(query=query)
    bindings = outputs["results"]["bindings"]
    subclass_codes = []
    for binding in bindings:
        code = binding["subclass"]["value"].split("/")[-1]
        subclass_codes.append(code)
    return subclass_codes


WIKIDATA_ENDPOINT = "https://query.wikidata.org/sparql"


@retry(
    wait=wait_random_exponential(multiplier=1, max=5),
    stop=stop_after_attempt(10),
)
def sparql_query(*, endpoint: str = WIKIDATA_ENDPOINT, query: str | bytes) -> dict:
    user_agent = f"OAK-User-Agent/{sys.version_info[0]}.{sys.version_info[1]}"
    sparql = SPARQLWrapper(endpoint, agent=user_agent)
    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)
    sparql.setTimeout(300)
    result: dict = sparql.query().convert()  # type: ignore
    return result


WIKIDATA_PROPERTY_DIRECT_PREFIX = "http://www.wikidata.org/prop/direct/"
