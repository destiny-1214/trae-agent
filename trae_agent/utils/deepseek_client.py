# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

"""Deepseek client wrapper with tool integrations"""

from .config import ModelParameters
from .models.deepseek import DeepseekProvider
from .models.openai_compatible_base import OpenAICompatibleClient


class DeepseekClient(OpenAICompatibleClient):
    """Deepseek client wrapper that maintains compatibility while using the new architecture."""

    def __init__(self, model_parameters: ModelParameters):
        super().__init__(model_parameters, DeepseekProvider())
