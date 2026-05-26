from dataclasses import dataclass
from enum import Enum, auto

from okgc.utils.sparql import Predicate, TypeInfo


class TransformKind(Enum):
    SWAP_SUBJECT_OBJECT = auto()
    ADD_SUBJECT_TYPE = auto()
    ADD_OBJECT_TYPE = auto()
    REPLACE_PREDICATE = auto()
    REPLACE_SUBJECT = auto()
    REPLACE_OBJECT = auto()


@dataclass(frozen=True)
class Transform:
    kind: TransformKind
    entity_type: TypeInfo | None = None
    predicate: Predicate | None = None
    entity: str | None = None

    def __repr__(self) -> str:
        if self.kind == TransformKind.SWAP_SUBJECT_OBJECT:
            return f"SwapSubjectObject"
        elif self.kind == TransformKind.ADD_SUBJECT_TYPE:
            assert self.entity_type is not None
            return f"AddSubjectType({self.entity_type.name})"
        elif self.kind == TransformKind.ADD_OBJECT_TYPE:
            assert self.entity_type is not None
            return f"AddObjectType({self.entity_type.name})"
        elif self.kind == TransformKind.REPLACE_PREDICATE:
            assert self.predicate is not None
            return f"ReplacePredicate({self.predicate.name})"
        elif self.kind == TransformKind.REPLACE_SUBJECT:
            assert self.entity is not None
            return f"ReplaceSubject({self.entity})"
        elif self.kind == TransformKind.REPLACE_OBJECT:
            assert self.entity is not None
            return f"ReplaceObject({self.entity})"
        assert False


@dataclass
class TransformCounters:
    num_gave_up: int = 0
    no_action: int = 0
    swap_subject_object: int = 0
    add_subject_type: int = 0
    add_object_type: int = 0
    replace_predicate: int = 0
    replace_subject: int = 0
    replace_object: int = 0
    num_retracts: int = 0

    def __add__(self, other: "TransformCounters") -> "TransformCounters":
        return TransformCounters(
            self.num_gave_up + other.num_gave_up,
            self.no_action + other.no_action,
            self.swap_subject_object + other.swap_subject_object,
            self.add_subject_type + other.add_subject_type,
            self.add_object_type + other.add_object_type,
            self.replace_predicate + other.replace_predicate,
            self.replace_subject + other.replace_subject,
            self.replace_object + other.replace_object,
            self.num_retracts + other.num_retracts,
        )
