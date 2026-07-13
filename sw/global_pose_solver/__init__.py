"""Global Pose Graph Solver for multi-part CAD assembly.

Takes pairwise relative pose constraints from JoinABLe (or any pair solver)
and finds globally consistent SE(3) poses for all parts via non-linear least
squares optimization.  No hard-coded case logic — purely graph-driven.
"""

from .audited_hypothesis_solver import solve_bounded_global_pose


def build_pools_from_joinable_reports(*args, **kwargs):
    """Lazy compatibility entry point for the JoinABLe-specific adapter.

    The manifold solver and residual provider are deliberately usable in a
    lightweight CAD/OCCT runtime.  Importing the package must not eagerly pull
    in the legacy JoinABLe/SDF dependency stack merely to access those generic
    components.
    """
    from .joinable_adapter import build_pools_from_joinable_reports as impl
    return impl(*args, **kwargs)


def make_occt_exact_validator(*args, **kwargs):
    """Lazy compatibility entry point for the OCCT JoinABLe adapter."""
    from .joinable_adapter import make_occt_exact_validator as impl
    return impl(*args, **kwargs)


def __getattr__(name):
    if name == "MultiBodyContactRefiner":
        from .contact_refinement import MultiBodyContactRefiner
        return MultiBodyContactRefiner
    raise AttributeError(name)

__all__ = [
    "solve_bounded_global_pose",
    "build_pools_from_joinable_reports",
    "make_occt_exact_validator",
    "MultiBodyContactRefiner",
]
