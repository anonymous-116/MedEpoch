from __future__ import annotations

import asyncio

from amf.providers.base import BaseProvider, ProviderRequest, ProviderResponse

class ProviderRegistry:
    def __init__(self) -> None:
        self._providers: dict[str, BaseProvider] = {}

    def register(self, provider: BaseProvider) -> None:
        if provider.name in self._providers:
            raise ValueError(f"Provider already registered: {provider.name}")
        self._providers[provider.name] = provider

    def get(self, name: str) -> BaseProvider:
        if name not in self._providers:
            raise KeyError(f"Provider not registered: {name}")
        return self._providers[name]

    def has(self, name: str) -> bool:
        return name in self._providers

    async def call(self, request: ProviderRequest) -> ProviderResponse:
        if request.provider not in self._providers:
            return ProviderResponse(ok=False, error=f"Provider not registered: {request.provider}")
        provider = self._providers[request.provider]
        try:
            return await asyncio.wait_for(provider.call(request), timeout=request.timeout)
        except TimeoutError:
            return ProviderResponse(ok=False, error=f"Provider timed out: {request.provider}")
        except Exception as exc:
            return ProviderResponse(ok=False, error=f"Provider error: {exc}")
