"""Frozen vision backbones with a common token output."""

from semtrace.backbones.base import BackboneOutput, TinyBackbone
from semtrace.backbones.dinov3 import DINOv3Backbone

__all__ = ["BackboneOutput", "DINOv3Backbone", "TinyBackbone"]
