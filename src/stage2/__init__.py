"""Paper DiT with its internal-guidance base head."""
from .models import Stage2ModelProtocol
from .models.DDT import DiTwDDTHeadIG

__all__ = ["DiTwDDTHeadIG", "Stage2ModelProtocol"]
