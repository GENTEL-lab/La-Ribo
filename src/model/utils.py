import math

import torch
from torch.nn import functional as F


def get_index_embedding(indices, embedding_dim, max_len=2056):

    K = torch.arange(embedding_dim // 2, device=indices.device)
    pos_embedding_sin = torch.sin(
        indices[..., None] * math.pi / (max_len ** (2 * K[None] / embedding_dim))
    ).to(indices.device)
    pos_embedding_cos = torch.cos(
        indices[..., None] * math.pi / (max_len ** (2 * K[None] / embedding_dim))
    ).to(indices.device)
    pos_embedding = torch.cat([pos_embedding_sin, pos_embedding_cos], dim=-1)
    return pos_embedding


def get_timestep_embedding(timesteps, embedding_dim, max_positions=10000):

    assert len(timesteps.shape) == 1
    timesteps = timesteps * max_positions
    half_dim = embedding_dim // 2
    embedding = math.log(max_positions) / (half_dim - 1)
    embedding = torch.exp(
        torch.arange(half_dim, dtype=torch.float32, device=timesteps.device)
        * -embedding
    )
    embedding = timesteps.float()[:, None] * embedding[None, :]
    embedding = torch.cat([torch.sin(embedding), torch.cos(embedding)], dim=1)
    if embedding_dim % 2 == 1:
        embedding = F.pad(embedding, (0, 1), mode="constant")
    assert embedding.shape == (timesteps.shape[0], embedding_dim)
    return embedding


def distance_to_distogram(dists, min_bin, max_bin, num_bins, *, dtype=None):

    dtype = dists.dtype if dtype is None else dtype
    lower = torch.linspace(min_bin, max_bin, num_bins, device=dists.device, dtype=dtype)
    upper = torch.cat([lower[1:], lower.new_tensor([1e8])], dim=-1)
    dists = dists.to(dtype)[..., None]
    return ((dists > lower) * (dists < upper)).to(dtype)


def calc_distogram(pos, min_bin, max_bin, num_bins):
    dists_2d = torch.linalg.norm(pos[:, :, None, :] - pos[:, None, :, :], axis=-1)
    return distance_to_distogram(dists_2d, min_bin, max_bin, num_bins, dtype=pos.dtype)
