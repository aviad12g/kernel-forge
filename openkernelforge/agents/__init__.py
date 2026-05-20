"""Candidate-generation agent interfaces."""

from openkernelforge.agents.base import CandidateSpec, KernelAgent
from openkernelforge.agents.backends import FakeBackend, ModelBackend
from openkernelforge.agents.dummy_agent import DummyAgent
from openkernelforge.agents.llm_agent import LLMAgent

__all__ = ["CandidateSpec", "DummyAgent", "FakeBackend", "KernelAgent", "LLMAgent", "ModelBackend"]
