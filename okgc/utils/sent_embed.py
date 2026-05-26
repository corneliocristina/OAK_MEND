import math

import torch

from okgc.utils.openai_wrapper import OpenAIClient


def _fast_cosine_similarity(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-9):
    """torch.cosine_similarity requires a lot of memory due to broadcasting since it generalizes to batched tensors.
    Here, we do not use broadcasting but rather a matmul.
    Inspired from the discussion here: https://github.com/pytorch/pytorch/issues/104564
    """

    if len(x.shape) != 2 or len(y.shape) != 2 or x.shape[1] != y.shape[1]:
        raise ValueError(
            f"Expected inputs to be of shape (M, K) and (N, K), but found {x.shape} and {y.shape}, respectively"
        )
    if x.requires_grad or y.requires_grad:
        raise ValueError(
            f"Expected inputs to not require gradients, but found {x.requires_grad} and {y.requires_grad}"
        )

    x_norm = torch.linalg.vector_norm(x, dim=1, keepdims=True)
    y_norm = torch.linalg.vector_norm(y, dim=1, keepdims=True)
    x_norm.clamp_(math.sqrt(eps))
    y_norm.clamp_(math.sqrt(eps))
    x = x / x_norm
    y = y / y_norm
    return torch.mm(x, y.T)


class SentenceEmbedder:
    def __init__(self, client: OpenAIClient):
        super().__init__()
        self.client = client

    def encode(
        self, ss: str | list[str], embed_size: int | None = None
    ) -> torch.Tensor:
        if isinstance(ss, str):
            ss = [ss]
        embeddings, usage_info = self.client.get_embeddings(ss, embed_size=embed_size)
        return torch.tensor(embeddings)

    def similarity(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return _fast_cosine_similarity(x, y)
