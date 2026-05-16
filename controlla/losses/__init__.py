"""Loss definitions for Controlla."""

from .contrastive_loss import ContrastiveLoss
from .emotion_loss import EmotionConsistencyLoss
from .graph_loss import GraphConsistencyLoss
from .identity_loss import IdentityPreservationLoss
from .total_loss import ControllaLoss

__all__ = [
    "ContrastiveLoss",
    "EmotionConsistencyLoss",
    "GraphConsistencyLoss",
    "IdentityPreservationLoss",
    "ControllaLoss",
]