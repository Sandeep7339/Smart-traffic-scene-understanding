"""High-level 3D box estimation approaches.

- geometric: purely geometry/depth fitting
- learned: neural RGB-D fusion model
"""

from . import geometric, learned

__all__ = ["geometric", "learned"]
