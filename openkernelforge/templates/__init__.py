"""Deterministic Triton template candidates."""

from openkernelforge.templates.elementwise_templates import (
    TemplateVariant,
    generate_elementwise_templates,
    render_bias_relu_template,
    render_relu_template,
    render_vector_add_template,
)
from openkernelforge.templates.template_agent import TemplateAgent

__all__ = [
    "TemplateAgent",
    "TemplateVariant",
    "generate_elementwise_templates",
    "render_bias_relu_template",
    "render_relu_template",
    "render_vector_add_template",
]
