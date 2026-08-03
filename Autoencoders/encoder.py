"""
Encoder module for asymmetric autoencoder.
"""

import torch
import torch.nn as nn
from typing import List, Optional


ACTIVATIONS = {
    "relu": nn.ReLU,
    "leaky_relu": nn.LeakyReLU,
    "elu": nn.ELU,
    "selu": nn.SELU,
    "tanh": nn.Tanh,
    "sigmoid": nn.Sigmoid,
    "gelu": nn.GELU,
    "none": nn.Identity,
}


def build_mlp_block(
    in_features: int,
    out_features: int,
    activation: str = "relu",
    dropout: float = 0.0,
    batch_norm: bool = False,
    layer_norm: bool = False,
    activation_kwargs: Optional[dict] = None,
) -> nn.Sequential:
    """Builds a single linear block: Linear -> [Norm] -> Activation -> [Dropout]."""
    if activation not in ACTIVATIONS:
        raise ValueError(f"Unknown activation '{activation}'. Choose from: {list(ACTIVATIONS.keys())}")

    layers = [nn.Linear(in_features, out_features)]

    if batch_norm:
        layers.append(nn.BatchNorm1d(out_features))
    elif layer_norm:
        layers.append(nn.LayerNorm(out_features))

    act_cls = ACTIVATIONS[activation]
    layers.append(act_cls(**(activation_kwargs or {})))

    if dropout > 0.0:
        layers.append(nn.Dropout(p=dropout))

    return nn.Sequential(*layers)


class Encoder(nn.Module):
    """
    Fully parametrizable MLP encoder.

    Builds a stack of hidden layers followed by a projection to the latent space.
    Each hidden layer can have its own activation, norm, and dropout settings.

    Args:
        input_dim:          Dimensionality of the input features.
        latent_dim:         Dimensionality of the latent space (bottleneck).
        hidden_dims:        List of hidden layer sizes. Empty list => direct input->latent.
        activation:         Activation for all hidden layers (or list, one per layer).
        latent_activation:  Activation applied after the latent projection ('none' = linear).
        dropout:            Dropout probability (shared, or list one per hidden layer).
        batch_norm:         Use BatchNorm after each hidden linear layer.
        layer_norm:         Use LayerNorm after each hidden linear layer (mutually exclusive with batch_norm).
        activation_kwargs:  Extra kwargs forwarded to the activation constructor.
    """

    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        hidden_dims: List[int] = None,
        activation: str | List[str] = "relu",
        latent_activation: str = "none",
        dropout: float | List[float] = 0.0,
        batch_norm: bool = False,
        layer_norm: bool = False,
        activation_kwargs: Optional[dict] = None,
    ):
        super().__init__()

        hidden_dims = hidden_dims or []
        n_hidden = len(hidden_dims)

        # Normalise activation / dropout to per-layer lists
        activations = (
            activation if isinstance(activation, list) else [activation] * n_hidden
        )
        dropouts = (
            dropout if isinstance(dropout, list) else [dropout] * n_hidden
        )

        if len(activations) != n_hidden:
            raise ValueError(
                f"len(activation)={len(activations)} must match len(hidden_dims)={n_hidden}"
            )
        if len(dropouts) != n_hidden:
            raise ValueError(
                f"len(dropout)={len(dropouts)} must match len(hidden_dims)={n_hidden}"
            )

        # Build hidden layers
        dims = [input_dim] + hidden_dims
        hidden_layers = []
        for i in range(n_hidden):
            hidden_layers.append(
                build_mlp_block(
                    in_features=dims[i],
                    out_features=dims[i + 1],
                    activation=activations[i],
                    dropout=dropouts[i],
                    batch_norm=batch_norm,
                    layer_norm=layer_norm,
                    activation_kwargs=activation_kwargs,
                )
            )

        self.hidden = nn.Sequential(*hidden_layers)

        # Latent projection
        latent_in = hidden_dims[-1] if hidden_dims else input_dim
        self.latent_proj = nn.Linear(latent_in, latent_dim)
        self.latent_act = ACTIVATIONS[latent_activation]()

        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.hidden(x)
        z = self.latent_act(self.latent_proj(h))
        return z

    def extra_repr(self) -> str:
        return (
            f"input_dim={self.input_dim}, "
            f"hidden_dims={self.hidden_dims}, "
            f"latent_dim={self.latent_dim}"
        )
