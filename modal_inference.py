"""
Convenience Modal entrypoint.

The main implementation lives in `simulation/modal_inference.py` to keep Modal-specific code isolated.
Deploy either file with:
  modal deploy simulation/modal_inference.py
or
  modal deploy modal_inference.py
"""

from simulation.modal_inference import app, agent_inference, persona_param_inference  # noqa: F401

