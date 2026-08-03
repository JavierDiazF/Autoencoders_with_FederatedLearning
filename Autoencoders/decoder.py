"""
Decoder module for asymmetric autoencoder.

Intentionally independent from the encoder: it can have a completely different
number of layers, widths, activations, and regularisation settings.
"""

import torch
import torch.nn as nn
from typing import List, Optional

from encoder import ACTIVATIONS, build_mlp_block


class Decoder(nn.Module):
    """
    Fully parametrizable MLP decoder.

    Mirror image of the Encoder in spirit, but with entirely independent
    architecture choices. The output activation lets you constrain the
    reconstruction range (e.g. 'sigmoid' for [0,1] data, 'none' for raw floats).

    Args:
        latent_dim:         Dimensionality of the latent space (input to decoder).
        output_dim:         Dimensionality of the reconstructed output.
        hidden_dims:        List of hidden layer sizes. Empty list => direct latent->output.
        activation:         Activation for all hidden layers (or list, one per layer).
        output_activation:  Activation applied on the final reconstruction ('none' = linear).
        dropout:            Dropout probability (shared, or list one per hidden layer).
        batch_norm:         Use BatchNorm after each hidden linear layer.
        layer_norm:         Use LayerNorm after each hidden linear layer.
        activation_kwargs:  Extra kwargs forwarded to the hidden activation constructor.
    """

    def __init__(
        self,
        latent_dim: int,
        output_dim: int,
        hidden_dims: List[int] = None,
        activation: str | List[str] = "relu",
        output_activation: str = "none",
        dropout: float | List[float] = 0.0,
        batch_norm: bool = False,
        layer_norm: bool = False,
        activation_kwargs: Optional[dict] = None,
    ):
        super().__init__()

        hidden_dims = hidden_dims or []
        n_hidden = len(hidden_dims)

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

        dims = [latent_dim] + hidden_dims
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

        # Output projection + activation
        out_in = hidden_dims[-1] if hidden_dims else latent_dim
        self.output_proj = nn.Linear(out_in, output_dim)
        self.output_act = ACTIVATIONS[output_activation]()

        self.latent_dim = latent_dim
        self.output_dim = output_dim
        self.hidden_dims = hidden_dims

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = self.hidden(z)
        x_hat = self.output_act(self.output_proj(h))
        return x_hat

    def extra_repr(self) -> str:
        return (
            f"latent_dim={self.latent_dim}, "
            f"hidden_dims={self.hidden_dims}, "
            f"output_dim={self.output_dim}"
        )
