import json
import os
from collections import defaultdict

from okgc.datasets.utils import DataEntry


def load_hotpot(
    filepath: str,
    *,
    datapath: str = "data",
) -> list[DataEntry]:
    filepath = os.path.join(datapath, filepath)
    with open(filepath, "r") as fp:
        data = json.load(fp)

    entries: list[DataEntry] = []
    for row in data:
        titles = [t for t, _ in row["context"]]
        texts = ["".join(ss) for _, ss in row["context"]]
        context: dict[str, list[str]] = {t: ctx for t, ctx in row["context"]}
        supporting_facts: dict[str, list[str]] = defaultdict(list)
        for t, j in row["supporting_facts"]:
            supporting_facts[t].append(context[t][j])
        entry = DataEntry(
            texts=texts,
            labels={
                "id": row["_id"],
                "type": row["type"],
                "level": row["level"],
            },
            extra_info={
                "text_idx": {t: i for i, t in enumerate(titles)},
                "question": row["question"],
                "answer": row["answer"],
                "supporting_facts": supporting_facts,
            },
        )
        entries.append(entry)
    return entries


def load_hotpot1000(datapath: str = "data") -> list[DataEntry]:
    return load_hotpot(os.path.join("hotpot1000", "hotpot1000.json"), datapath=datapath)
