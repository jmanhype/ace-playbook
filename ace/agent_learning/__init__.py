"""Agent Learning (Early Experience) integration for ACE.

This package hosts the components that previously lived in the separate
``AgentLearningEE`` repository.  They provide a light-weight agent training loop
that can stream experience back into the ACE playbook using the generator →
reflector → curator pipeline exposed by :mod:`ace.runtime`.

The modules are deliberately small and composable so that production workflows
can wire in custom schedulers, replay buffers, or telemetry hooks while the
tests and quick-start script rely on the in-memory defaults.
"""

from ace.agent_learning.datasets import StaticTaskDataset
from ace.agent_learning.exploration import EpisodeResult, ExperienceBuffer
from ace.agent_learning.guardrails import PredictionGuardrails
from ace.agent_learning.live_loop import LiveLoop
from ace.agent_learning.policy import BasePolicy, EpsilonGreedyPolicy
from ace.agent_learning.reflection import InsightSection, ReflectionEngine, RuntimeReflector
from ace.agent_learning.types import FeedbackPacket, TaskSpecification, WorldModelPrediction
from ace.agent_learning.utils import create_in_memory_runtime, prepare_default_metrics
from ace.agent_learning.world_model import WorldModel

__all__ = [
    "BasePolicy",
    "EpsilonGreedyPolicy",
    "EpisodeResult",
    "ExperienceBuffer",
    "FeedbackPacket",
    "InsightSection",
    "LiveLoop",
    "PredictionGuardrails",
    "ReflectionEngine",
    "RuntimeReflector",
    "StaticTaskDataset",
    "TaskSpecification",
    "WorldModel",
    "WorldModelPrediction",
    "create_in_memory_runtime",
    "prepare_default_metrics",
]
