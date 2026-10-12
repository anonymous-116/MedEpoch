from __future__ import annotations

import asyncio
import copy
import json
import os
import urllib.error
import urllib.request
from abc import ABC, abstractmethod
from typing import Any, Awaitable, Callable, Sequence, TypeVar

from amf.llm.input_truncation import HeadTailMessageTruncator
from amf.llm.usage import LLMUsageTracker

T = TypeVar("T")

FALLBACK_CONTENT = "I don't know."
FALLBACK_STATUS_CODES = {400, 408, 429, 500, 502, 503, 504}
DATA_INSPECTION_ERROR_CODE = "data_inspection_failed"
DATA_INSPECTION_ERROR_MESSAGES = (
    "Input text data may contain inappropriate content.",
    "Output data may contain inappropriate content.",
)

def exception_status_code(exc: Exception) -> int | None:
    """Extract an HTTP status from SDK, urllib, or wrapped exceptions."""

    current: BaseException | None = exc
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        status = getattr(current, "status_code", None)
        if isinstance(status, int):
            return status
        response = getattr(current, "response", None)
        status = getattr(response, "status_code", None)
        if isinstance(status, int):
            return status
        code = getattr(current, "code", None)
        if isinstance(code, int):
            return code
        current = current.__cause__ or current.__context__
    return None

def exception_error_code(exc: Exception) -> str | None:
    """Extract a provider error code, including from wrapped exceptions."""

    current: BaseException | None = exc
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        code = getattr(current, "code", None)
        if isinstance(code, str):
            return code
        body = getattr(current, "body", None)
        if isinstance(body, dict):
            error = body.get("error", body)
            if isinstance(error, dict) and isinstance(error.get("code"), str):
                return error["code"]
        current = current.__cause__ or current.__context__
    return None

def is_data_inspection_failure(exc: Exception) -> bool:
    """Recognize a DashScope inspection rejection from any known signal.

    Some SDK error variants omit the HTTP status, and output inspection uses a
    different message from input inspection.  Any one of the provider's known
    status, error code, or messages is therefore enough to activate the
    context-appending retry path.
    """

    message = str(exc).lower()
    return (
        exception_status_code(exc) == 400
        or exception_error_code(exc) == DATA_INSPECTION_ERROR_CODE
        or any(marker.lower() in message for marker in DATA_INSPECTION_ERROR_MESSAGES)
    )

def run_with_retries(
    operation: Callable[[], T],
    *,
    max_retries: int,
    retry_delay_seconds: float,
    prepare_retry: Callable[[Exception, int, bool], bool] | None = None,
) -> T:
    """Run a sync operation with a fixed maximum number of attempts."""

    attempts = max(1, max_retries)
    last_error: Exception | None = None
    for attempt in range(attempts):
        try:
            return operation()
        except Exception as exc:
            last_error = exc
            attempt_number = attempt + 1
            will_retry = attempt_number < attempts
            retry_prepared = bool(
                prepare_retry
                and prepare_retry(exc, attempt_number, will_retry)
            )
            status_code = exception_status_code(exc)
            error_code = exception_error_code(exc)
            error_details = str(exc).replace("\n", " ").strip()
            if retry_prepared:
                retry_message = "retrying immediately with appended data-inspection context"
            else:
                retry_message = (
                    f"retrying in {retry_delay_seconds:g}s"
                    if will_retry
                    else "no attempts remaining"
                )
            print(
                "[LLM request failed] "
                f"attempt={attempt_number}/{attempts} "
                f"error_type={type(exc).__name__} "
                f"status_code={status_code} "
                f"error_code={error_code} "
                f"message={error_details} "
                f"next={retry_message}",
                flush=True,
            )
            if not will_retry:
                break
            if not retry_prepared and retry_delay_seconds > 0:
                import time

                time.sleep(retry_delay_seconds)
    assert last_error is not None
    raise last_error

async def run_async_with_retries(
    operation: Callable[[], Awaitable[T]],
    *,
    max_retries: int,
    retry_delay_seconds: float,
    attempt_timeout_seconds: float,
    prepare_retry: Callable[[Exception, int, bool], bool] | None = None,
) -> T:
    """Run an async operation with retries and a hard limit per attempt."""

    attempts = max(1, max_retries)
    last_error: Exception | None = None
    for attempt in range(attempts):
        try:
            async with asyncio.timeout(attempt_timeout_seconds):
                return await operation()
        except Exception as exc:
            last_error = exc
            attempt_number = attempt + 1
            will_retry = attempt_number < attempts
            retry_prepared = bool(
                prepare_retry
                and prepare_retry(exc, attempt_number, will_retry)
            )
            status_code = exception_status_code(exc)
            error_code = exception_error_code(exc)
            error_details = str(exc).replace("\n", " ").strip()
            if retry_prepared:
                retry_message = "retrying immediately with appended data-inspection context"
            else:
                retry_message = (
                    f"retrying in {retry_delay_seconds:g}s"
                    if will_retry
                    else "no attempts remaining"
                )
            print(
                "[LLM request failed] "
                f"attempt={attempt_number}/{attempts} "
                f"error_type={type(exc).__name__} "
                f"status_code={status_code} "
                f"error_code={error_code} "
                f"message={error_details} "
                f"next={retry_message}",
                flush=True,
            )
            if not will_retry:
                break
            if not retry_prepared and retry_delay_seconds > 0:
                await asyncio.sleep(retry_delay_seconds)
    assert last_error is not None
    raise last_error

class BaseLLMClient(ABC):
    def __init__(
        self,
        *,
        usage_tracker: LLMUsageTracker | None = None,
        usage_role: str = "unknown",
    ) -> None:
        self.usage_tracker = usage_tracker
        self.usage_role = usage_role
        self.last_usage: dict[str, Any] | None = None

        self.last_request_recovery: dict[str, Any] | None = None
        self.last_input_truncation: dict[str, Any] | None = None
        self.last_finish_reason: str | None = None

        self.last_request_messages: list[dict[str, Any]] | None = None

    def fork(self) -> "BaseLLMClient":
        """Create request-local mutable state while sharing immutable config.

        Concurrent episodes may use the same run-level UsageTracker, whose
        provenance is already task-local via ContextVar.  Fields such as
        ``last_usage`` and recovery/reasoning audit data must not be shared,
        because an Agent reads them immediately after its own request.
        """

        client = copy.copy(self)
        client.last_usage = None
        client.last_request_recovery = None
        client.last_input_truncation = None
        client.last_finish_reason = None
        client.last_request_messages = None
        if hasattr(client, "last_reasoning_content"):
            client.last_reasoning_content = ""
        return client

    def _data_inspection_retry_preparer(
        self,
        *,
        prompt: str,
        request_kwargs: dict[str, Any],
        context_message: str | None,
        provider: str,
        model: str,
        enabled: bool = True,
    ) -> tuple[
        Callable[[Exception, int, bool], bool] | None,
        dict[str, Any] | None,
    ]:
        """Build a retry hook that appends context after inspection rejection."""

        if not enabled or not context_message:
            return None, None

        recovery = {
            "type": "data_inspection_context_retry",
            "trigger_count": 0,
            "context_append_count": 0,
            "context_message": context_message,
            "failed_attempts": [],
        }

        def prepare_retry(exc: Exception, attempt_number: int, will_retry: bool) -> bool:
            if not is_data_inspection_failure(exc):
                return False

            recovery["trigger_count"] += 1
            recovery["failed_attempts"].append(attempt_number)
            appended = False
            if will_retry:
                messages = request_kwargs.get("messages")
                if messages is None:
                    messages = [{"role": "user", "content": prompt}]
                request_kwargs["messages"] = [
                    *list(messages),
                    {"role": "user", "content": context_message},
                ]
                recovery["context_append_count"] += 1
                appended = True

            self._record_usage(
                provider=provider,
                model=model,
                usage={
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                },
                metadata={
                    "request_failed": True,
                    "upstream_error": {
                        "status_code": exception_status_code(exc),
                        "error_code": exception_error_code(exc),
                        "error_type": type(exc).__name__,
                        "message": str(exc)[:1000],
                    },
                    "request_recovery": {
                        **recovery,
                        "context_appended_for_next_attempt": appended,
                    },
                },
            )
            return appended

        return prepare_retry, recovery

    def _finish_request_recovery(
        self,
        recovery: dict[str, Any] | None,
        *,
        recovered: bool,
    ) -> None:
        """Expose a compact recovery audit to both Usage and trajectory data."""

        if not recovery or not recovery.get("trigger_count"):
            self.last_request_recovery = None
            return
        recovery["recovered"] = recovered
        self.last_request_recovery = dict(recovery)
        if self.last_usage is not None:
            self.last_usage["request_recovery"] = dict(recovery)

    def _record_usage(
        self,
        *,
        provider: str,
        model: str,
        usage: Any,
        response_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        if self.usage_tracker is None:
            self.last_usage = None
            return None
        effective_metadata = dict(metadata or {})
        if self.last_input_truncation is not None:
            effective_metadata["input_truncation"] = dict(self.last_input_truncation)
        self.last_usage = self.usage_tracker.record(
            role=self.usage_role,
            provider=provider,
            model=model,
            usage=usage,
            response_id=response_id,
            metadata=effective_metadata or None,
        )
        return self.last_usage

    def _fallback_for_exception(
        self,
        exc: Exception,
        *,
        provider: str,
        model: str,
    ) -> str:
        status_code = exception_status_code(exc)
        if status_code is not None and status_code not in FALLBACK_STATUS_CODES:
            raise exc
        error = {
            "status_code": status_code,
            "error_code": exception_error_code(exc),
            "error_type": type(exc).__name__,
            "message": str(exc)[:1000],
        }

        self._record_usage(
            provider=provider,
            model=model,
            usage={
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
            },
            metadata={
                "fallback": True,
                "fallback_content": FALLBACK_CONTENT,
                "upstream_error": error,
            },
        )
        return FALLBACK_CONTENT

    @abstractmethod
    async def generate(self, prompt: str = "", **kwargs: Any) -> str:
        """Generate a model response from either a prompt or chat messages."""

class ScriptedLLMClient(BaseLLMClient):
    """Deterministic LLM client for tests and scripted runs."""

    def __init__(
        self,
        responses: Sequence[str],
        *,
        usage_tracker: LLMUsageTracker | None = None,
        usage_role: str = "scripted",
    ):
        super().__init__(usage_tracker=usage_tracker, usage_role=usage_role)
        if not responses:
            raise ValueError("ScriptedLLMClient requires at least one response.")
        self._responses = list(responses)
        self._index = 0

    async def generate(self, prompt: str = "", **kwargs: Any) -> str:
        self.last_request_recovery = None
        self.last_request_messages = [
            dict(message)
            for message in kwargs.get(
                "messages",
                [{"role": "user", "content": prompt}],
            )
        ]
        self._record_usage(provider="scripted", model="scripted", usage=None)
        if self._index >= len(self._responses):
            return self._responses[-1]
        response = self._responses[self._index]
        self._index += 1
        return response

def adapt_response_format(
    response_format: dict[str, Any] | None,
    json_schema_support: bool,
) -> dict[str, Any] | None:
    """Translate strict json_schema requests for endpoints without support.

    DashScope endpoints accept {"type": "json_object"} but reject
    {"type": "json_schema"} with 400 invalid_parameter_error ("This
    response_format type is unavailable now"). The callers' prompts already
    describe the expected JSON shape and parse content JSON themselves, so
    json_object keeps the same contract without schema enforcement.
    """
    if response_format is None or json_schema_support:
        return response_format
    if response_format.get("type") == "json_schema":
        return {"type": "json_object"}
    return response_format

class OpenAICompatibleLLMClient(BaseLLMClient):
    """Minimal `/v1/chat/completions` client for OpenAI-compatible APIs."""

    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        api_key_env: str = "OPENAI_API_KEY",
        base_url: str = "https://api.openai.com/v1",
        timeout: int = 60,
        temperature: float = 0.0,
        top_p: float | None = None,
        seed: int = 42,
        enable_thinking: bool | None = None,
        json_schema_support: bool = True,
        max_completion_tokens: int | None = None,
        data_inspection_retry_enabled: bool = True,
        max_retries: int = 3,
        retry_delay_seconds: float = 10.0,
        fallback_on_error: bool = True,
        usage_tracker: LLMUsageTracker | None = None,
        usage_role: str = "agent",
    ) -> None:
        super().__init__(usage_tracker=usage_tracker, usage_role=usage_role)
        self.model = model
        self.api_key = api_key or os.getenv(api_key_env)
        self.api_key_env = api_key_env
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.temperature = temperature
        self.top_p = top_p
        self.seed = seed
        self.enable_thinking = enable_thinking
        self.json_schema_support = json_schema_support
        self.max_completion_tokens = max_completion_tokens
        self.data_inspection_retry_enabled = data_inspection_retry_enabled
        self.max_retries = max_retries
        self.retry_delay_seconds = retry_delay_seconds
        self.fallback_on_error = fallback_on_error

    async def generate(self, prompt: str = "", **kwargs: Any) -> str:
        if not self.api_key:
            raise RuntimeError(f"Missing API key. Set {self.api_key_env}.")
        self.last_usage = None
        self.last_request_recovery = None
        self.last_request_messages = None
        request_kwargs = dict(kwargs)
        context_message = request_kwargs.pop("data_inspection_context", None)
        model = str(request_kwargs.get("model", self.model))
        prepare_retry, recovery = self._data_inspection_retry_preparer(
            prompt=prompt,
            request_kwargs=request_kwargs,
            context_message=str(context_message) if context_message else None,
            provider="openai_compatible",
            model=model,
            enabled=self.data_inspection_retry_enabled,
        )
        try:
            result = await asyncio.to_thread(
                run_with_retries,
                lambda: self._generate_sync(prompt, **request_kwargs),
                max_retries=int(request_kwargs.get("max_retries", self.max_retries)),
                retry_delay_seconds=float(
                    request_kwargs.get("retry_delay_seconds", self.retry_delay_seconds)
                ),
                prepare_retry=prepare_retry,
            )
        except Exception as exc:
            self._finish_request_recovery(recovery, recovered=False)
            if request_kwargs.get("fallback_on_error", self.fallback_on_error):
                return self._fallback_for_exception(
                    exc,
                    provider="openai_compatible",
                    model=model,
                )
            raise
        self._finish_request_recovery(recovery, recovered=True)
        return result

    def _generate_sync(self, prompt: str = "", **kwargs: Any) -> str:
        url = f"{self.base_url}/chat/completions"
        messages = [
            dict(message)
            for message in kwargs.get(
                "messages",
                [{"role": "user", "content": prompt}],
            )
        ]
        self.last_request_messages = messages
        body = {
            "model": kwargs.get("model", self.model),
            "messages": messages,
            "temperature": kwargs.get("temperature", self.temperature),
            "seed": int(kwargs.get("seed", self.seed)),
        }
        thinking = kwargs.get("enable_thinking", self.enable_thinking)
        if thinking is not None:

            body["enable_thinking"] = bool(thinking)
        top_p = kwargs.get("top_p", self.top_p)
        if top_p is not None:

            body["top_p"] = float(top_p)
        max_completion_tokens = kwargs.get(
            "max_completion_tokens",
            self.max_completion_tokens,
        )
        if max_completion_tokens is not None:
            body["max_completion_tokens"] = int(max_completion_tokens)
        if kwargs.get("stop") is not None:
            body["stop"] = kwargs["stop"]
        if kwargs.get("response_format") is not None:
            body["response_format"] = adapt_response_format(
                kwargs["response_format"], self.json_schema_support
            )

        request = urllib.request.Request(
            url=url,
            data=json.dumps(body).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"LLM HTTP error {exc.code}: {detail}") from exc

        choices = payload.get("choices") or []
        if not choices:
            raise RuntimeError(f"LLM response contains no choices: {payload}")
        message = choices[0].get("message") or {}
        content = message.get("content")
        if content is None:
            raise RuntimeError(f"LLM response contains no message content: {payload}")
        self._record_usage(
            provider="openai_compatible",
            model=str(body["model"]),
            usage=payload.get("usage"),
            response_id=payload.get("id"),
        )
        return str(content)

class OpenAISDKLLMClient(BaseLLMClient):
    """OpenAI Python SDK client for OpenAI-compatible providers.

    This class matches providers that expose the OpenAI SDK surface, including
    DashScope-compatible endpoints. Streaming responses return only final
    `delta.content`; optional reasoning chunks are stored on
    `last_reasoning_content` for inspection.
    """

    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        api_key_env: str = "OPENAI_API_KEY",
        base_url: str = "https://api.openai.com/v1",
        timeout: int = 60,
        temperature: float | None = None,
        top_p: float | None = None,
        seed: int = 42,
        stream: bool = True,
        enable_thinking: bool | None = None,
        json_schema_support: bool = True,
        extra_body: dict[str, Any] | None = None,
        max_completion_tokens: int | None = None,
        input_token_limit: int | None = None,
        input_tokenizer_file: str | None = None,
        input_template_reserve_tokens: int = 512,
        system_only_prompt_as_user: bool = False,
        fallback_system_prompt: str = "You are a helpful assistant.",
        data_inspection_retry_enabled: bool = True,
        max_retries: int = 3,
        retry_delay_seconds: float = 10.0,
        include_usage: bool = True,
        fallback_on_error: bool = True,
        usage_tracker: LLMUsageTracker | None = None,
        usage_role: str = "agent",
    ) -> None:
        super().__init__(usage_tracker=usage_tracker, usage_role=usage_role)
        self.model = model
        self.api_key = api_key or os.getenv(api_key_env)
        self.api_key_env = api_key_env
        self.base_url = base_url
        self.timeout = timeout
        self.temperature = temperature
        self.top_p = top_p
        self.seed = seed
        self.stream = stream
        self.enable_thinking = enable_thinking
        self.json_schema_support = json_schema_support
        self.extra_body = extra_body or {}
        self.max_completion_tokens = max_completion_tokens
        self.input_truncator = (
            HeadTailMessageTruncator(
                tokenizer_file=input_tokenizer_file,
                max_input_tokens=input_token_limit,
                template_reserve_tokens=input_template_reserve_tokens,
            )
            if input_token_limit is not None and input_tokenizer_file
            else None
        )

        self.system_only_prompt_as_user = system_only_prompt_as_user
        self.fallback_system_prompt = fallback_system_prompt
        self.data_inspection_retry_enabled = data_inspection_retry_enabled
        self.last_reasoning_content = ""
        self.max_retries = max_retries
        self.retry_delay_seconds = retry_delay_seconds
        self.include_usage = include_usage
        self.fallback_on_error = fallback_on_error

    async def generate(self, prompt: str = "", **kwargs: Any) -> str:
        if not self.api_key:
            raise RuntimeError(f"Missing API key. Set {self.api_key_env}.")
        self.last_usage = None
        self.last_request_recovery = None
        self.last_input_truncation = None
        self.last_request_messages = None
        request_kwargs = dict(kwargs)
        context_message = request_kwargs.pop("data_inspection_context", None)
        attempt_timeout = float(request_kwargs.get("timeout", self.timeout))
        model = str(request_kwargs.get("model", self.model))
        prepare_retry, recovery = self._data_inspection_retry_preparer(
            prompt=prompt,
            request_kwargs=request_kwargs,
            context_message=str(context_message) if context_message else None,
            provider="openai_sdk",
            model=model,
            enabled=self.data_inspection_retry_enabled,
        )
        try:
            result = await run_async_with_retries(
                lambda: self._generate_async(prompt, **request_kwargs),
                max_retries=int(request_kwargs.get("max_retries", self.max_retries)),
                retry_delay_seconds=float(
                    request_kwargs.get("retry_delay_seconds", self.retry_delay_seconds)
                ),
                attempt_timeout_seconds=attempt_timeout,
                prepare_retry=prepare_retry,
            )
        except Exception as exc:
            self._finish_request_recovery(recovery, recovered=False)
            if request_kwargs.get("fallback_on_error", self.fallback_on_error):
                return self._fallback_for_exception(
                    exc,
                    provider="openai_sdk",
                    model=model,
                )
            raise
        self._finish_request_recovery(recovery, recovered=True)
        return result

    async def _generate_async(self, prompt: str = "", **kwargs: Any) -> str:
        try:
            from openai import AsyncOpenAI
        except ImportError as exc:
            raise RuntimeError("OpenAI SDK is not installed. Run `pip install openai`.") from exc

        async with AsyncOpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=kwargs.get("timeout", self.timeout),
            max_retries=0,
        ) as client:
            request = self._build_request(prompt, kwargs)
            completion = await client.chat.completions.create(**request)
            if request["stream"]:
                return await self._collect_stream_async(
                    completion,
                    model=str(request["model"]),
                )
            return self._collect_completion(completion, model=str(request["model"]))

    def _generate_sync(self, prompt: str = "", **kwargs: Any) -> str:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("OpenAI SDK is not installed. Run `pip install openai`.") from exc

        client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=kwargs.get("timeout", self.timeout),

            max_retries=0,
        )
        request = self._build_request(prompt, kwargs)
        completion = client.chat.completions.create(**request)
        if request["stream"]:
            return self._collect_stream(completion, model=str(request["model"]))
        return self._collect_completion(completion, model=str(request["model"]))

    def _build_request(
        self,
        prompt: str,
        kwargs: dict[str, Any],
    ) -> dict[str, Any]:
        messages = [
            dict(message)
            for message in kwargs.get(
                "messages",
                [{"role": "user", "content": prompt}],
            )
        ]
        if (
            self.system_only_prompt_as_user
            and len(messages) == 1
            and messages[0].get("role") == "system"
        ):
            assembled_prompt = messages[0].get("content", "")
            messages = [
                {"role": "system", "content": self.fallback_system_prompt},
                {"role": "user", "content": assembled_prompt},
            ]
        if self.input_truncator is not None:
            messages, truncation = self.input_truncator.truncate(list(messages))
            self.last_input_truncation = truncation
            if truncation:
                print(
                    "[LLM input truncated] "
                    f"original_content_tokens={truncation['original_content_tokens']} "
                    f"final_content_tokens={truncation['final_content_tokens']} "
                    f"limit={truncation['max_input_tokens']}",
                    flush=True,
                )
        self.last_request_messages = [dict(message) for message in messages]
        extra_body = dict(self.extra_body)
        thinking = kwargs.get("enable_thinking", self.enable_thinking)
        if thinking is not None:

            extra_body["enable_thinking"] = bool(thinking)

        request: dict[str, Any] = {
            "model": kwargs.get("model", self.model),
            "messages": messages,
            "stream": bool(kwargs.get("stream", self.stream)),
            "seed": int(kwargs.get("seed", self.seed)),
        }
        temperature = kwargs.get("temperature", self.temperature)
        if temperature is not None:
            request["temperature"] = temperature
        top_p = kwargs.get("top_p", self.top_p)
        if top_p is not None:
            request["top_p"] = float(top_p)
        if kwargs.get("stop") is not None:
            request["stop"] = kwargs["stop"]
        if kwargs.get("response_format") is not None:

            request["response_format"] = adapt_response_format(
                kwargs["response_format"], self.json_schema_support
            )
        max_completion_tokens = kwargs.get(
            "max_completion_tokens",
            self.max_completion_tokens,
        )
        if max_completion_tokens is not None:
            request["max_completion_tokens"] = int(max_completion_tokens)
        if extra_body:
            request["extra_body"] = extra_body

        if request["stream"] and kwargs.get("include_usage", self.include_usage):
            request["stream_options"] = {"include_usage": True}

        return request

    def _collect_stream(self, completion: Any, *, model: str) -> str:
        self.last_finish_reason = None
        content_parts: list[str] = []
        reasoning_parts: list[str] = []
        usage = None
        response_id = None
        for chunk in completion:
            chunk_usage = getattr(chunk, "usage", None)
            if chunk_usage is not None:
                usage = chunk_usage
            response_id = response_id or getattr(chunk, "id", None)
            choices = getattr(chunk, "choices", None) or []
            if not choices:
                continue
            self.last_finish_reason = getattr(choices[0], "finish_reason", None) or self.last_finish_reason
            delta = getattr(choices[0], "delta", None)
            if delta is None:
                continue
            reasoning = getattr(delta, "reasoning_content", None)
            if reasoning is None:

                reasoning = getattr(delta, "reasoning", None)
            if reasoning is not None:
                reasoning_parts.append(str(reasoning))
            content = getattr(delta, "content", None)
            if content:
                content_parts.append(str(content))
        self.last_reasoning_content = "".join(reasoning_parts)
        self._record_usage(
            provider="openai_sdk",
            model=model,
            usage=usage,
            response_id=response_id,
        )
        return "".join(content_parts)

    async def _collect_stream_async(
        self,
        completion: Any,
        *,
        model: str,
    ) -> str:
        self.last_finish_reason = None
        content_parts: list[str] = []
        reasoning_parts: list[str] = []
        usage = None
        response_id = None
        async for chunk in completion:
            chunk_usage = getattr(chunk, "usage", None)
            if chunk_usage is not None:
                usage = chunk_usage
            response_id = response_id or getattr(chunk, "id", None)
            choices = getattr(chunk, "choices", None) or []
            if not choices:
                continue
            self.last_finish_reason = getattr(choices[0], "finish_reason", None) or self.last_finish_reason
            delta = getattr(choices[0], "delta", None)
            if delta is None:
                continue
            reasoning = getattr(delta, "reasoning_content", None)
            if reasoning is None:
                reasoning = getattr(delta, "reasoning", None)
            if reasoning is not None:
                reasoning_parts.append(str(reasoning))
            content = getattr(delta, "content", None)
            if content:
                content_parts.append(str(content))
        self.last_reasoning_content = "".join(reasoning_parts)
        self._record_usage(
            provider="openai_sdk",
            model=model,
            usage=usage,
            response_id=response_id,
        )
        return "".join(content_parts)

    def _collect_completion(self, completion: Any, *, model: str) -> str:
        choices = getattr(completion, "choices", None) or []
        if not choices:
            raise RuntimeError(f"LLM response contains no choices: {completion}")
        self.last_finish_reason = getattr(choices[0], "finish_reason", None)
        message = getattr(choices[0], "message", None)
        if message is None:
            raise RuntimeError(f"LLM response contains no message: {completion}")
        reasoning = getattr(message, "reasoning_content", None)
        if reasoning is None:
            reasoning = getattr(message, "reasoning", None)
        self.last_reasoning_content = str(reasoning or "")
        content = getattr(message, "content", None)
        if content is None:
            raise RuntimeError(f"LLM response contains no message content: {completion}")
        self._record_usage(
            provider="openai_sdk",
            model=model,
            usage=getattr(completion, "usage", None),
            response_id=getattr(completion, "id", None),
        )
        return str(content)
