"""
Asymmetric Autoencoder: combines an independent Encoder and Decoder.

The encoder and decoder can have completely different depths, widths,
activations, and regularisation — hence 'asymmetric'.
"""

import torch
import torch.nn as nn
from typing import List, Optional

from encoder import Encoder
from decoder import Decoder


class AsymmetricAutoencoder(nn.Module):
    """
    Asymmetric Autoencoder for tabular data.

    Accepts pre-built Encoder/Decoder instances so each component can be
    configured and reused independently.

    Args:
        encoder: An Encoder instance.
        decoder: A Decoder instance.
        channel: Optional module applied to the latent code before decoding
            (e.g. PowerNormalization + AWGNChannel from channel.py). This is
            what turns a plain reconstruction AE into the encoder -> channel
            -> decoder communication AE described in Fig. 1b / 2a of the
            survey. Left as None, behaviour is unchanged (direct
            encoder -> decoder), which is what the IoT example uses.
    """

    def __init__(self, encoder: Encoder, decoder: Decoder, channel: Optional[nn.Module] = None):
        super().__init__()

        if encoder.latent_dim != decoder.latent_dim:
            raise ValueError(
                f"encoder.latent_dim ({encoder.latent_dim}) must match "
                f"decoder.latent_dim ({decoder.latent_dim})"
            )

        self.encoder = encoder
        self.decoder = decoder
        self.channel = channel

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Maps input to latent representation."""
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Maps latent representation back to input space."""
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (reconstruction, latent_code). Latent code is pre-channel."""
        z = self.encode(x)
        y = self.channel(z) if self.channel is not None else z
        x_hat = self.decode(y)
        return x_hat, z

    @property
    def input_dim(self) -> int:
        return self.encoder.input_dim

    @property
    def latent_dim(self) -> int:
        return self.encoder.latent_dim

    @property
    def output_dim(self) -> int:
        return self.decoder.output_dim

    def extra_repr(self) -> str:
        return (
            f"input_dim={self.input_dim}, "
            f"latent_dim={self.latent_dim}, "
            f"output_dim={self.output_dim}"
        )


# ── Factory helpers ──────────────────────────────────────────────────────────

def build_autoencoder(
    input_dim: int,
    latent_dim: int,
    encoder_hidden_dims: List[int] = None,
    decoder_hidden_dims: List[int] = None,
    encoder_activation: str | List[str] = "relu",
    decoder_activation: str | List[str] = "relu",
    encoder_latent_activation: str = "none",
    decoder_output_activation: str = "none",
    encoder_dropout: float | List[float] = 0.0,
    decoder_dropout: float | List[float] = 0.0,
    encoder_batch_norm: bool = False,
    decoder_batch_norm: bool = False,
    encoder_layer_norm: bool = False,
    decoder_layer_norm: bool = False,
    output_dim: Optional[int] = None,
    channel: Optional[nn.Module] = None,
) -> AsymmetricAutoencoder:
    """
    Convenience factory that builds an AsymmetricAutoencoder from flat parameters.

    If output_dim is None, it defaults to input_dim (standard reconstruction).
    If channel is None, encoder output feeds the decoder directly (IoT-style
    compression). Pass a channel module (see channel.py) to get the
    encoder -> channel -> decoder communication AE from the survey baseline.

    Example — deep encoder, shallow decoder:
        model = build_autoencoder(
            input_dim=128,
            latent_dim=16,
            encoder_hidden_dims=[256, 128, 64],
            decoder_hidden_dims=[64],
            encoder_activation="leaky_relu",
            decoder_activation="relu",
            encoder_dropout=0.2,
        )
    """
    output_dim = output_dim or input_dim

    encoder = Encoder(
        input_dim=input_dim,
        latent_dim=latent_dim,
        hidden_dims=encoder_hidden_dims,
        activation=encoder_activation,
        latent_activation=encoder_latent_activation,
        dropout=encoder_dropout,
        batch_norm=encoder_batch_norm,
        layer_norm=encoder_layer_norm,
    )

    decoder = Decoder(
        latent_dim=latent_dim,
        output_dim=output_dim,
        hidden_dims=decoder_hidden_dims,
        activation=decoder_activation,
        output_activation=decoder_output_activation,
        dropout=decoder_dropout,
        batch_norm=decoder_batch_norm,
        layer_norm=decoder_layer_norm,
    )

    return AsymmetricAutoencoder(encoder, decoder, channel=channel)
