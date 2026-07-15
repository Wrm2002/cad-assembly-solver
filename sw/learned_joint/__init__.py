"""Pure B-Rep joint hypotheses and constraint-manifold pose solving.

The package deliberately separates three concerns:

* Fusion 360 supervision is converted into a leakage-audited contract;
* learned JoinABLe entity-pair scores are lifted to geometric manifolds;
* multi-part poses are solved jointly without reading names or case labels.
"""

from .hypothesis import (
    attach_pose_initials,
    JointHypothesis,
    build_joint_hypotheses,
    frame_from_entity,
)
from .precision_pose_validator import PrecisionTolerances, validate_precision_pose

def solve_manifold_pose_graph(*args, **kwargs):
    """Lazily import the optional mesh/SDF global solver.

    Pair-level Pose-head training only needs ``pose_learning``.  Importing the
    global solver eagerly used to make that independent training path require
    ``pysdf`` even though no mesh residual is evaluated.  Keeping the import
    here preserves the public API for global-pose callers while isolating the
    optional dependency to the stage that genuinely uses it.
    """

    from .manifold_solver import solve_manifold_pose_graph as implementation
    return implementation(*args, **kwargs)

__all__ = [
    "JointHypothesis",
    "attach_pose_initials",
    "build_joint_hypotheses",
    "frame_from_entity",
    "PrecisionTolerances",
    "solve_manifold_pose_graph",
    "validate_precision_pose",
]
