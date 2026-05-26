import argparse
import os

from okgc.datasets.loaders import DATASET_NAMES, load_dataset
from okgc.utils.openai_wrapper import OpenAIClient
from okgc.utils.sent_embed import SentenceEmbedder
from okgc.utils.vector_index import StrIndex

parser = argparse.ArgumentParser(prog="Index for the vector RAG baseline")
parser.add_argument("dataset", choices=DATASET_NAMES)
parser.add_argument(
    "--sent-embed-api-base",
    help="The sentence embedding model API url including the port",
    default="http://localhost:9200/v1",
)
parser.add_argument(
    "--sent-embed",
    help="The sentence embedding model to use. If empty, then it defaults to the first model available on the endpoint",
    default="",
)
parser.add_argument(
    "--api-key", help="The API key", default="okgc-reflect-key-deadbeef"
)


if __name__ == "__main__":
    args = parser.parse_args()

    # Load the data
    data = load_dataset(args.dataset)

    # Retrieve the list of all text documents
    documents = [text for entry in data for text in entry.texts]

    # Setup the sentence embedder
    sent_embed = SentenceEmbedder(
        client=OpenAIClient(
            args.sent_embed, base_url=args.sent_embed_api_base, api_key=args.api_key
        )
    )

    # Create and serialize the index
    index_filename = f"vector_rag_{args.dataset}_index.pt"
    index = StrIndex(
        documents,
        filename=index_filename,
        sent_embed=sent_embed,
    )
