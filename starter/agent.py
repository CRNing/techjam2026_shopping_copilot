# Backwards-compat shim: the local evaluator (evaluator.py) imports
# `from starter.agent import Agent`, while the submission rules require a
# single top-level `agent.py` exporting `Agent`. Both paths point at the
# same implementation to avoid duplicated/diverging logic.
from agent import Agent  # noqa: F401