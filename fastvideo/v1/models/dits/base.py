# SPDX-License-Identifier: Apache-2.0

from torch import nn


# TODO
class BaseDiT(nn.Module):
    _fsdp_shard_conditions: list = []
    attention_head_dim: int | None = None
    _supported_attention_backends: list[str] = []

    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        if not self.supported_attention_backends:
            raise ValueError(
                f"Subclass {self.__class__.__name__} must define _supported_attention_backends"
            )

    def forward(self, *args, **kwargs):
        pass

    @property
    def supported_attention_backends(self) -> list[str]:
        return self._supported_attention_backends
