"""Compatibility import for the inference-side patch-result artifact.

The implementation remains outside the geometric preprocessing core; this
module exists only for callers that namespace all pipeline artifacts under
``preprocessing``.
"""

from patch_result_writer import *  # noqa: F401,F403
