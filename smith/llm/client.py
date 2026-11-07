"""Multi-provider LLM client with API key rotation, retries, and metrics."""
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence
import urllib.error
import urllib.request

from dotenv import load_dotenv

from smith.models import StepMetrics

DEFAULT_CONFIG_PATH = (
    Path(__file__).resolve().parents[2] / "configs" / "providers.json"
)

# Longest wait honoured on a 429, so a fallback still has time to answer.
MAX_RETRY_AFTER_SECONDS = 10.0

URL_TO_ENV: Dict[str, str] = {
    "openrouter.ai": "OPENROUTER_API_KEY",
    "api.groq.com": "GROQ_API_KEY",
    "api.cerebras.ai": "CEREBRAS_API_KEY",
    "api.mistral.ai": "MISTRAL_API_KEY",
    "generativelanguage.googleapis.com": "GEMINI_API_KEY",
    "api.together.xyz": "TOGETHER_API_KEY",
}


class LLMError(RuntimeError):
    """Raised when all LLM API attempts and retries fail."""


class KeyRotator:
    """Round-robin API key rotator loading keys from environment/.env."""

    def __init__(
        self, env_var: str, explicit_keys: Optional[Sequence[str]] = None
    ) -> None:
        load_dotenv()
        self.env_var = env_var
        self.keys: List[str] = (
            [k.strip() for k in explicit_keys if k.strip()]
            if explicit_keys
            else self._discover_keys(env_var)
        )
        self._index = 0

    @staticmethod
    def _discover_keys(env_var: str) -> List[str]:
        found: List[str] = []
        raw = os.environ.get(env_var, "")
        for part in raw.split(","):
            clean = part.strip()
            if clean and clean not in found:
                found.append(clean)
        prefix = f"{env_var}_"
        for k in sorted(os.environ.keys()):
            if k.startswith(prefix):
                for part in os.environ[k].split(","):
                    clean = part.strip()
                    if clean and clean not in found:
                        found.append(clean)
        return found

    def current(self) -> str:
        """Return the current active API key."""
        if not self.keys:
            raise LLMError(
                f"No API keys found in environment variable '{self.env_var}'."
            )
        return self.keys[self._index % len(self.keys)]

    def rotate(self) -> str:
        """Advance to the next API key and return it."""
        if not self.keys:
            raise LLMError(
                f"No API keys found in environment variable '{self.env_var}'."
            )
        self._index = (self._index + 1) % len(self.keys)
        return self.keys[self._index]


class LLMClient:
    """OpenAI-compatible LLM client with token rotation and StepMetrics."""

    def __init__(
        self,
        model_name: Optional[str] = None,
        provider_url: Optional[str] = None,
        api_keys: Optional[Sequence[str]] = None,
        max_retries: int = 4,
        timeout_seconds: float = 60.0,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        config_path: Optional[Path] = None,
        fallback: Optional["LLMClient"] = None,
    ) -> None:
        load_dotenv()
        self.fallback = fallback
        cfg = self._load_config(config_path or DEFAULT_CONFIG_PATH)
        default_prov = cfg.get("default_provider", "mistral")
        prov_map = cfg.get("providers", {})

        self.api_url = (
            provider_url
            or os.environ.get("LLM_PROVIDER_URL")
            or prov_map.get(default_prov, {}).get(
                "api_url", "https://api.mistral.ai/v1"
            )
        ).rstrip("/")
        self.model_name = (
            model_name
            or os.environ.get("LLM_MODEL_NAME")
            or cfg.get("default_model", "codestral-latest")
        )
        env_var = self._resolve_env_var(self.api_url, prov_map)
        self.rotator = KeyRotator(env_var, explicit_keys=api_keys)
        self.max_retries = max(0, max_retries)
        self.timeout_seconds = timeout_seconds
        self.temperature = temperature
        self.max_tokens = max_tokens

    @staticmethod
    def _load_config(path: Path) -> Dict[str, Any]:
        if path.is_file():
            try:
                data: Dict[str, Any] = json.loads(
                    path.read_text(encoding="utf-8")
                )
                return data
            except Exception:
                return {}
        return {}

    @staticmethod
    def _resolve_env_var(
        api_url: str, prov_map: Dict[str, Any]
    ) -> str:
        for info in prov_map.values():
            if info.get("api_url", "").rstrip("/") == api_url:
                env_name: str = info.get("env_var", "OPENROUTER_API_KEY")
                return env_name
        for domain, env_var in URL_TO_ENV.items():
            if domain in api_url:
                return env_var
        return "LLM_API_KEY"

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        return max(1, len(text) // 4) if text else 0

    def _post_chat(
        self, payload: Dict[str, Any], api_key: str, timeout: float
    ) -> Dict[str, Any]:
        endpoint = f"{self.api_url}/chat/completions"
        body = json.dumps(payload).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
            "User-Agent": "Agent-Smith/0.1",
        }
        req = urllib.request.Request(
            endpoint, data=body, headers=headers, method="POST"
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
            parsed: Dict[str, Any] = json.loads(raw)
            return parsed

    def generate(
        self,
        messages: List[Dict[str, str]],
        step: int = 1,
        stop: Optional[List[str]] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        timeout: Optional[float] = None,
    ) -> StepMetrics:
        """Call the LLM, then the fallback client if this one fails.

        `timeout` caps the whole call in seconds, fallback included.
        """
        start = time.perf_counter()
        try:
            return self._generate_once(
                messages, step, stop, max_tokens, temperature, timeout)
        except LLMError as exc:
            left = None if timeout is None else (
                timeout - (time.perf_counter() - start))
            if self.fallback is None or (left is not None and left <= 0):
                raise
            try:
                return self.fallback.generate(
                    messages, step, stop, max_tokens, temperature, left)
            except LLMError as fallback_exc:
                raise LLMError(
                    f"{exc}; fallback {self.fallback.model_name} failed "
                    f"too: {fallback_exc}") from fallback_exc

    def _generate_once(
        self,
        messages: List[Dict[str, str]],
        step: int = 1,
        stop: Optional[List[str]] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        timeout: Optional[float] = None,
    ) -> StepMetrics:
        """Call the LLM chat completion API and return a populated StepMetrics.

        The caller (agent loop) only needs to fill in `sandbox_input` and
        `sandbox_output` after executing the extracted code.
        `timeout` caps the whole call in seconds, retries included.
        """
        payload: Dict[str, Any] = {
            "model": self.model_name,
            "messages": messages,
            "temperature": (
                self.temperature if temperature is None else temperature
            ),
            "max_tokens": (
                self.max_tokens if max_tokens is None else max_tokens
            ),
        }
        if stop:
            payload["stop"] = stop

        retries = 0
        last_error: Optional[Exception] = None
        start_time = time.perf_counter()
        deadline = None if timeout is None else start_time + timeout

        for attempt in range(self.max_retries + 1):
            request_timeout = self.timeout_seconds
            if deadline is not None:
                request_timeout = min(
                    request_timeout, deadline - time.perf_counter()
                )
                if request_timeout <= 0:
                    break
            api_key = self.rotator.current()
            try:
                data = self._post_chat(payload, api_key, request_timeout)
                elapsed_ms = round(
                    (time.perf_counter() - start_time) * 1000.0, 2
                )
                content = self._extract_content(data)
                in_tok, out_tok = self._extract_usage(
                    data, messages, content
                )
                return StepMetrics(
                    step=step,
                    input_tokens=in_tok,
                    output_tokens=out_tok,
                    request_time_ms=elapsed_ms,
                    api_url=self.api_url,
                    model_name=self.model_name,
                    llm_output=content,
                    retries=retries,
                )
            except urllib.error.HTTPError as exc:
                detail = ""
                try:
                    detail = exc.read().decode("utf-8", errors="replace")[:200]
                except Exception:
                    pass
                last_error = RuntimeError(
                    f"{exc} ({self.api_url} {self.model_name}): {detail}"
                )
                self.rotator.rotate()
                if attempt < self.max_retries:
                    retries += 1
                    self._sleep_backoff(
                        attempt, self._retry_after(exc, deadline))
            except Exception as exc:
                last_error = exc
                self.rotator.rotate()
                if attempt < self.max_retries:
                    retries += 1
                    self._sleep_backoff(attempt)

        raise LLMError(
            f"LLM request failed after {retries} retries: {last_error}"
        )

    @staticmethod
    def _sleep_backoff(attempt: int, wait: float = 0.0) -> None:
        """Exponential backoff, or the provider's wait when longer."""
        time.sleep(max(wait, min(4.0, 0.5 * (2 ** attempt))))

    @staticmethod
    def _retry_after(
        exc: urllib.error.HTTPError, deadline: Optional[float]
    ) -> float:
        """Seconds a rate-limited provider asks to wait, within the budget.

        The 4s backoff was too short for per-minute quotas such as Groq's.
        """
        if exc.code != 429:
            return 0.0
        # Without Retry-After (Mistral sends none), wait for a per-minute
        # window to free up: 4 retries of 10s cover a 2 requests/min quota.
        try:
            wait = float(exc.headers.get("retry-after", ""))
        except (TypeError, ValueError):
            wait = MAX_RETRY_AFTER_SECONDS
        if deadline is not None:
            wait = min(wait, deadline - time.perf_counter())
        return max(0.0, min(wait, MAX_RETRY_AFTER_SECONDS))

    @staticmethod
    def _extract_content(data: Dict[str, Any]) -> str:
        choices = data.get("choices") or []
        if not choices:
            return ""
        msg = choices[0].get("message") or {}
        content = msg.get("content")
        if isinstance(content, str):
            return content
        return str(choices[0].get("text") or "")

    def _extract_usage(
        self,
        data: Dict[str, Any],
        messages: List[Dict[str, str]],
        content: str,
    ) -> tuple[int, int]:
        usage = data.get("usage") or {}
        in_tok = int(usage.get("prompt_tokens") or 0)
        out_tok = int(usage.get("completion_tokens") or 0)
        if in_tok <= 0:
            prompt_text = " ".join(m.get("content", "") for m in messages)
            in_tok = self._estimate_tokens(prompt_text)
        if out_tok <= 0:
            out_tok = self._estimate_tokens(content)
        return in_tok, out_tok
