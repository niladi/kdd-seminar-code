from typing import Optional
from torch import Tensor


from dataclasses import dataclass


@dataclass
class ModelResult:
    logits: Tensor
    loss: Optional[Tensor]
