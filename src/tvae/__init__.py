"""Student-t VAEs through a shared Gaussian-Gamma scale mixture."""

from .priors import PriorTerms, StudentTPrior
from .models import ForwardOutput, LossOutput, TVAE

__version__ = "0.1.0"
__all__ = ["StudentTPrior", "PriorTerms", "TVAE", "ForwardOutput", "LossOutput"]
