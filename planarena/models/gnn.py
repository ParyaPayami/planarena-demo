"""Relational graph neural network over the parcel graph.

Two heads share a context embedding:
* suitability: P(land use | zoning, lot, neighbourhood). Trained by masked-node
  modelling: a random subset of parcels has its own use hidden and the model
  must recover it from context.
* value: log assessed value per lot sq ft, given the context embedding and an
  explicit land use. Swapping the use input gives a counterfactual estimate.

The embedding is always computed with the target parcel's own use masked, so
both heads condition only on context.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, SAGEConv

from planarena.config import USE_KEYS
from planarena.models.features import context_features, value_target, zone_family  # noqa: F401

RELATIONS = ("adjacent", "near", "same_zoning")
N_USES = len(USE_KEYS)


class ParcelGNN(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 128, layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.inp = nn.Linear(in_dim + N_USES + 1, hidden)
        self.convs = nn.ModuleList([
            HeteroConv({("parcel", r, "parcel"): SAGEConv(hidden, hidden) for r in RELATIONS}, aggr="sum")
            for _ in range(layers)
        ])
        self.norms = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(layers)])
        self.dropout = dropout
        self.suit_head = nn.Linear(hidden, N_USES)
        self.value_head = nn.Sequential(nn.Linear(hidden + N_USES, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def embed(self, x, use, masked, edge_index_dict):
        """``use`` holds int land-use codes (-1 = unknown). ``masked`` is a bool tensor of hidden parcels."""
        oh = torch.zeros(x.shape[0], N_USES, device=x.device)
        visible = (~masked) & (use >= 0)
        oh[visible, use[visible]] = 1.0
        h = F.relu(self.inp(torch.cat([x, oh, masked.float().unsqueeze(1)], dim=1)))
        for conv, norm in zip(self.convs, self.norms):
            out = conv({"parcel": h}, edge_index_dict)["parcel"]
            h = norm(h + F.relu(out))
            h = F.dropout(h, self.dropout, self.training)
        return h

    def suitability_logits(self, h):
        return self.suit_head(h)

    def value(self, h, use_codes):
        oh = F.one_hot(use_codes.clamp(min=0), N_USES).float()
        return self.value_head(torch.cat([h, oh], dim=1)).squeeze(1)


def edge_dict(graph: dict, device="cpu") -> dict:
    return {("parcel", r, "parcel"): torch.as_tensor(graph[r], dtype=torch.long, device=device) for r in RELATIONS}


def attach_value_model(engine, processed=None) -> bool:
    """Use the trained GNN value head for counterfactual value deltas of changed parcels.

    Returns False (and leaves the rate-table value model in place) when no trained model exists.
    """
    import numpy as np

    from planarena.config import PROCESSED

    d = processed or PROCESSED
    if not (d / "gnn.pt").exists():
        return False
    ckpt = torch.load(d / "gnn.pt", map_location="cpu")
    model = ParcelGNN(ckpt["in_dim"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    ctx = torch.tensor(np.load(d / "gnn_context.npy"))
    lot = engine.lot

    @torch.no_grad()
    def value_fn(use, idx):
        i = torch.as_tensor(idx, dtype=torch.long)
        u = torch.as_tensor(np.clip(use[idx], 0, None), dtype=torch.long)
        return np.expm1(model.value(ctx[i], u).numpy()) * lot[idx]

    engine.value_model = value_fn
    return True
