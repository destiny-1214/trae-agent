# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

"""Deepseek provider configuration."""

import openai

from .openai_compatible_base import ProviderConfig


class DeepseekProvider(ProviderConfig):
    """Deepseek provider configuration."""

    def create_client(
        self, api_key: str, base_url: str | None, api_version: str | None
    ) -> openai.OpenAI:
        """Create OpenAI client with Deepseek base URL."""
        return openai.OpenAI(base_url=base_url, api_key=api_key)

    def get_service_name(self) -> str:
        """Get the service name for retry logging."""
        return "Deepseek"

    def get_provider_name(self) -> str:
        """Get the provider name for trajectory recording."""
        return "deepseek"

    def get_extra_headers(self) -> dict[str, str]:
        """Get Deepseek-specific headers (none needed)."""
        return {}

    def supports_tool_calling(self, model_name: str) -> bool:
        """Check if the model supports tool calling."""
        # Deepseek models generally support tool calling
        return True
