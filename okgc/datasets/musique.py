import json
import os

from okgc.datasets.utils import DataEntry


def load_musique(
    filepath: str,
    *,
    datapath: str = "data",
) -> list[DataEntry]:
    filepath = os.path.join(datapath, filepath)
    with open(filepath, "r") as fp:
        data = json.load(fp)

    entries: list[DataEntry] = []
    for row in data:
        titles = [x["title"] for x in row["paragraphs"]]
        texts = [x["paragraph_text"] for x in row["paragraphs"]]
        full_texts = [f"### {t}\n{s}" for t, s in zip(titles, texts)]
        entry = DataEntry(
            texts=full_texts,
            labels={
                "id": row["id"],
            },
            extra_info={
                "question": row["question"],
                "answer": row["answer"],
            },
        )
        entries.append(entry)
    return entries


def load_musique1000(datapath: str = "data") -> list[DataEntry]:
    return load_musique(
        os.path.join("musique1000", "musique1000.json"), datapath=datapath
    )
