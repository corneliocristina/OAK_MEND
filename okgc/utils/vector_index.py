import os
from collections import defaultdict

import torch

from okgc.utils.sent_embed import SentenceEmbedder
from okgc.utils.sparql import Predicate, TypeInfo, TypesHierarchy


class VectorIndex:
    def __init__(
        self,
        data: dict[str, tuple[str, ...]],
        *,
        sent_embed: SentenceEmbedder,
        cache_filename: str | None = None,
    ):
        if cache_filename is not None:
            cache_path = "vindex"
            os.makedirs(cache_path, exist_ok=True)
            self.cache_filepath = os.path.join(cache_path, cache_filename)
        else:
            self.cache_filepath = None
        #
        self.sent_embed = sent_embed

        # Build the index, if needed
        if self.cache_filepath is not None and os.path.isfile(self.cache_filepath):
            index_state = torch.load(self.cache_filepath)
        else:
            # We want to build a map embedding -> list of identifiers
            # E.g., embedding of a certain type alias -> list of type identifiers having that alias
            keys = list(data.keys())
            values = list(data[k] for k in keys)
            dedup_values: dict[str, list[str]] = defaultdict(list)
            for k, vs in zip(keys, values):
                for v in vs:
                    dedup_values[v].append(k)
            embedding_keys: list[str] = list(dedup_values.keys())
            embeddings: torch.Tensor = self.sent_embed.encode(embedding_keys)
            index: list[list[str]] = [
                dedup_values[dedup_k] for dedup_k in embedding_keys
            ]
            index_state = {"embeddings": embeddings, "index": index}
            if self.cache_filepath is not None:
                torch.save(index_state, self.cache_filepath)
        self.index_state = index_state

    @property
    def embeddings(self) -> torch.Tensor:
        return self.index_state["embeddings"]

    @property
    def index(self) -> list[list[str]]:
        return self.index_state["index"]

    def _search(
        self,
        q: str | list[str],
        k: int = 5,
        threshold: float | None = None,
        *,
        return_scores: bool = False,
    ) -> list[list[str]] | tuple[list[list[str]], list[list[float]]]:
        qs: list[str]
        if isinstance(q, str):
            qs = [q]
        else:
            qs = q
        q_embedding = self.sent_embed.encode(qs)
        scores = self.sent_embed.similarity(q_embedding, self.index_state["embeddings"])
        indices = torch.argsort(scores, dim=-1, descending=True).tolist()
        result_keys: list[list[str]] = []
        result_scores: list[list[float]] = []
        for r in range(len(qs)):
            rk: list[str] = []
            rs: list[float] = []
            for i in indices[r]:
                if len(rk) > k:
                    rk = rk[:k]
                    rs = rs[:k]
                    break
                retrieved = self.index_state["index"][i]
                score = scores[r, i].item()
                for j in retrieved:
                    if j in rk or (threshold is not None and score < threshold):
                        continue
                    rk.append(j)
                    rs.append(score)
            result_keys.append(rk)
            result_scores.append(rs)
        if return_scores:
            return result_keys, result_scores
        return result_keys


class TypesVectorIndex(VectorIndex):
    def __init__(
        self,
        hierarchy: TypesHierarchy,
        filename: str | None = "types_index.pt",
        **kwargs,
    ):
        data = {
            c: tuple([t.name] + (t.aliases if t.aliases is not None else []))
            for c, t in hierarchy.types.items()
        }
        super().__init__(data, cache_filename=filename, **kwargs)
        self.hierarchy = hierarchy

    def __contains__(self, code: str) -> bool:
        return code in self.hierarchy

    def __getitem__(self, code: str) -> TypeInfo:
        return self.hierarchy[code]

    def is_subclass(self, t1: TypeInfo, t2: TypeInfo) -> bool:
        return self.hierarchy.is_subclass(t1, t2)

    def search(
        self, q: str | list[str], k: int = 10, *, return_scores: bool = False
    ) -> list[list[TypeInfo]] | tuple[list[list[TypeInfo]], list[list[float]]]:
        output = self._search(q, k=k, return_scores=True)
        assert isinstance(output, tuple)
        codes, scores = output
        types = [[self.hierarchy[c] for c in cs] for cs in codes]
        if return_scores:
            return types, scores
        return types


class PredicatesVectorIndex(VectorIndex):
    def __init__(
        self,
        predicates: dict[str, Predicate],
        filename: str | None = "predicates_index.pt",
        **kwargs,
    ):
        data = {
            c: tuple([p.name] + (p.aliases if p.aliases is not None else []))
            for c, p in predicates.items()
        }
        super().__init__(data, cache_filename=filename, **kwargs)
        self.predicates = predicates

    def __contains__(self, code: str) -> bool:
        return code in self.predicates

    def __getitem__(self, code: str) -> Predicate:
        return self.predicates[code]

    def search(
        self, q: str | list[str], k: int = 10, *, return_scores: bool = False
    ) -> list[list[Predicate]] | tuple[list[list[Predicate]], list[list[float]]]:
        output = self._search(q, k=k, return_scores=True)
        assert isinstance(output, tuple)
        codes, scores = output
        predicates = [[self.predicates[c] for c in cs] for cs in codes]
        if return_scores:
            return predicates, scores
        return predicates


class StrIndex(VectorIndex):
    def __init__(
        self,
        ss: list[str] | dict[str, list[str]],
        filename: str | None = None,
        **kwargs,
    ):
        if isinstance(ss, list):
            data = {s: (s,) for s in ss}
        else:
            assert isinstance(ss, dict)
            data = {s: tuple(aliases) for s, aliases in ss.items()}
        super().__init__(data, cache_filename=filename, **kwargs)
        self.strings = ss

    def search(
        self,
        q: str | list[str],
        k: int = 10,
        threshold: float | None = None,
        *,
        return_scores: bool = False,
    ) -> list[list[str]] | tuple[list[list[str]], list[list[float]]]:
        output = self._search(q, k=k, threshold=threshold, return_scores=True)
        assert isinstance(output, tuple)
        strings, scores = output
        if return_scores:
            return strings, scores
        return strings
