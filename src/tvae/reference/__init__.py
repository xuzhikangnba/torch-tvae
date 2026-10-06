"""Original multi-component experiment models, with checkpoint-compatible names.

These preserve the archived loss and detached responsibilities. They are separate
from the single-component TVAE interface and do not imply benchmark replication.
"""
from .model28 import StudentTVaDE as TVaDE28
from .model128 import StudentTVaDE as TVaDE128

__all__ = ["TVaDE28", "TVaDE128"]
