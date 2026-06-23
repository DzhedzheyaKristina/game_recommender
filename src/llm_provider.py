"""Provider-specific helpers for real LLM calls and readiness checks."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
import json
import os
import uuid
from time import sleep

import requests
from openai import OpenAI

from src.config import Settings, normalize_llm_response_language


SUPPORTED_LLM_PROVIDERS = {"openai", "openrouter", "gigachat", "mock"}
DEFAULT_GIGACHAT_OAUTH_URL = "https://ngw.devices.sberbank.ru:9443/api/v2/oauth"
DEFAULT_GIGACHAT_API_BASE_URL = "https://gigachat.devices.sberbank.ru/api/v1"
DEFAULT_GIGACHAT_SCOPE = "GIGACHAT_API_PERS"
DEFAULT_GIGACHAT_MODEL = "GigaChat"
DEFAULT_GIGACHAT_TOKEN_TTL_MINUTES = 25


@dataclass(slots=True)
class CachedToken:
    """In-memory GigaChat token cache."""

    access_token: str
    expires_at: datetime
    obtained_at: datetime
    token_status: str


_GIGACHAT_TOKEN_CACHE: CachedToken | None = None


def normalize_llm_provider(value: object) -> str:
    """Normalize the selected LLM provider."""

    normalized = str(value or "openai").strip().lower() or "openai"
    if normalized not in SUPPORTED_LLM_PROVIDERS:
        raise ValueError(
            "LLM_PROVIDER must be one of: openai, openrouter, gigachat, mock."
        )
    return normalized


def get_effective_llm_provider(settings: Settings) -> str:
    """Return the normalized provider name."""

    return normalize_llm_provider(getattr(settings, "llm_provider", "openai"))


def get_effective_llm_model(settings: Settings) -> str:
    """Return the selected model for the current provider."""

    provider = get_effective_llm_provider(settings)
    if provider == "gigachat":
        return str(
            getattr(settings, "gigachat_model", None)
            or getattr(settings, "llm_model", None)
            or DEFAULT_GIGACHAT_MODEL
        ).strip() or DEFAULT_GIGACHAT_MODEL
    return str(getattr(settings, "llm_model", None) or "").strip()


def get_response_language(settings: Settings) -> str:
    """Return the configured response language."""

    return normalize_llm_response_language(getattr(settings, "llm_response_language", "ru"))


def get_gigachat_ca_bundle_path(settings: Settings | None = None) -> str:
    """Return the configured GigaChat CA bundle path, if any."""

    if settings is not None:
        candidate = str(getattr(settings, "gigachat_ca_bundle", "") or "").strip()
        if candidate:
            return candidate
    return str(os.getenv("GIGACHAT_CA_BUNDLE", "") or "").strip()


def inspect_gigachat_ca_bundle(settings: Settings | None = None) -> dict[str, object]:
    """Inspect the configured CA bundle without exposing its contents."""

    bundle_path = get_gigachat_ca_bundle_path(settings)
    exists = bool(bundle_path) and Path(bundle_path).is_file()
    certificate_count = 0
    if exists:
        try:
            bundle_text = Path(bundle_path).read_text(encoding="utf-8", errors="ignore")
        except OSError:
            bundle_text = ""
        certificate_count = bundle_text.count("BEGIN CERTIFICATE")
    return {
        "gigachat_ca_bundle": bundle_path,
        "gigachat_ca_bundle_exists": exists,
        "gigachat_ca_bundle_certificate_count": certificate_count,
    }


def get_gigachat_verify_value(settings: Settings) -> bool | str:
    """Return the `verify` value for requests made to GigaChat."""

    verify_ssl = bool(getattr(settings, "gigachat_verify_ssl", True))
    if not verify_ssl:
        print(
            "WARNING: GIGACHAT_VERIFY_SSL=false is intended only for local debugging and is not recommended."
        )
        return False

    bundle_path = get_gigachat_ca_bundle_path(settings)
    if bundle_path:
        bundle_file = Path(bundle_path)
        if not bundle_file.is_file():
            raise FileNotFoundError(f"GigaChat CA bundle file not found: {bundle_path}")
        return str(bundle_file)
    return True


def provider_credentials_configured(settings: Settings) -> bool:
    """Check whether the selected provider has usable credentials."""

    provider = get_effective_llm_provider(settings)
    model = get_effective_llm_model(settings)
    if provider == "mock":
        return True
    if provider == "gigachat":
        return bool(str(getattr(settings, "gigachat_auth_key", "") or "").strip() and model)
    if provider == "openrouter":
        openrouter_key = os.getenv("OPENROUTER_API_KEY", "").strip()
        if not openrouter_key:
            openrouter_key = str(getattr(settings, "openai_api_key", "") or "").strip()
        return bool(openrouter_key and model)
    return bool(str(getattr(settings, "openai_api_key", "") or "").strip() and model)


def generate_llm_json_response(
    provider: str,
    model: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float = 0.0,
    settings: Settings | None = None,
) -> str:
    """Call the selected provider and return the assistant content as text."""

    normalized_provider = normalize_llm_provider(provider)
    if normalized_provider == "mock":
        return json.dumps({"recommendations": []}, ensure_ascii=False)
    if normalized_provider in {"openai", "openrouter"}:
        return _generate_openai_compatible_response(
            provider=normalized_provider,
            model=model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=temperature,
        )
    if normalized_provider == "gigachat":
        return _generate_gigachat_response(
            model=model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=temperature,
            settings=settings,
        )
    raise ValueError(f"Unsupported LLM provider: {provider}")


def check_provider_readiness(settings: Settings) -> dict[str, object]:
    """Inspect provider configuration and optionally probe token retrieval."""

    provider = get_effective_llm_provider(settings)
    model = get_effective_llm_model(settings)
    response_language = get_response_language(settings)
    report: dict[str, object] = {
        "status": "missing_credentials",
        "selected_provider": provider,
        "selected_model": model,
        "response_language": response_language,
        "real_llm_calls_allowed": False,
        "client_initialized": False,
        "token_checked": False,
        "token_status": "",
        "error": "",
    }
    if provider == "mock":
        report.update(
            {
                "status": "configured",
                "real_llm_calls_allowed": False,
                "client_initialized": True,
            }
        )
        return report

    if provider == "gigachat":
        auth_key = str(getattr(settings, "gigachat_auth_key", "") or "").strip()
        scope = str(getattr(settings, "gigachat_scope", DEFAULT_GIGACHAT_SCOPE) or "").strip()
        verify_ssl = bool(getattr(settings, "gigachat_verify_ssl", True))
        bundle_report = inspect_gigachat_ca_bundle(settings)
        token_report = build_gigachat_token_preflight_report(settings, use_cache=False)
        report.update(
            {
                "gigachat_scope": scope,
                "gigachat_verify_ssl": verify_ssl,
                **bundle_report,
                **token_report,
            }
        )
        if not auth_key or not model:
            return report
        if bundle_report["gigachat_ca_bundle"] and not bundle_report["gigachat_ca_bundle_exists"]:
            report.update(
                {
                    "status": "ca_bundle_missing",
                    "token_checked": True,
                    "token_status": "ca_bundle_missing",
                    "error": f"GigaChat CA bundle file not found: {bundle_report['gigachat_ca_bundle']}",
                }
            )
            return report
        report["token_checked"] = True
        report["token_status"] = token_report.get("token_status", "token_error")
        report["error"] = token_report.get("error", "")
        if token_report.get("token_obtained", False):
            report.update(
                {
                    "status": "configured",
                    "real_llm_calls_allowed": True,
                    "client_initialized": True,
                }
            )
        else:
            report["status"] = token_report.get("token_status", "token_error")
        return report

    if provider in {"openai", "openrouter"}:
        api_key = (
            os.getenv("OPENROUTER_API_KEY", "").strip()
            if provider == "openrouter"
            else os.getenv("OPENAI_API_KEY", "").strip()
        )
        if not api_key or not model:
            return report
        try:
            client_kwargs: dict[str, object] = {"api_key": api_key}
            if provider == "openrouter":
                client_kwargs["base_url"] = os.getenv(
                    "OPENROUTER_API_BASE_URL", "https://openrouter.ai/api/v1"
                ).strip()
            OpenAI(**client_kwargs)
            report.update(
                {
                    "status": "configured",
                    "real_llm_calls_allowed": True,
                    "client_initialized": True,
                }
            )
            return report
        except Exception as exc:  # noqa: BLE001
            report["error"] = str(exc)
            report["status"] = "token_error"
            return report

    report["error"] = f"Unsupported provider: {provider}"
    return report


def _generate_openai_compatible_response(
    *,
    provider: str,
    model: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float,
) -> str:
    """Call an OpenAI-compatible chat-completions endpoint."""

    api_key = os.getenv("OPENROUTER_API_KEY", "").strip() if provider == "openrouter" else os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key and provider == "openrouter":
        api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("OpenAI-compatible API key is missing.")

    base_url = os.getenv("OPENROUTER_API_BASE_URL", "https://openrouter.ai/api/v1").strip()
    client_kwargs: dict[str, object] = {"api_key": api_key}
    if provider == "openrouter":
        client_kwargs["base_url"] = base_url
    client = OpenAI(**client_kwargs)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=temperature,
        response_format={"type": "json_object"},
    )
    return response.choices[0].message.content or ""


def _generate_gigachat_response(
    *,
    model: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float,
    settings: Settings | None = None,
) -> str:
    """Call the GigaChat chat-completions endpoint."""

    token = _get_gigachat_access_token(settings)
    api_base_url = (
        str(getattr(settings, "gigachat_api_base_url", DEFAULT_GIGACHAT_API_BASE_URL) or "").strip()
        if settings is not None
        else os.getenv("GIGACHAT_API_BASE_URL", DEFAULT_GIGACHAT_API_BASE_URL).strip()
    ) or DEFAULT_GIGACHAT_API_BASE_URL
    verify_value = (
        get_gigachat_verify_value(settings)
        if settings is not None
        else _resolve_gigachat_verify_without_settings()
    )

    response = requests.post(
        f"{api_base_url}/chat/completions",
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
        json={
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": temperature,
        },
        timeout=60,
        verify=verify_value,
    )
    response.raise_for_status()
    payload = response.json()
    choices = payload.get("choices") or []
    if not choices:
        raise RuntimeError("GigaChat response did not contain choices.")
    message = choices[0].get("message") or {}
    content = message.get("content", "")
    return str(content or "")


def _resolve_gigachat_verify_without_settings() -> bool | str:
    """Resolve the GigaChat verify flag from environment-only configuration."""

    verify_ssl = _parse_bool_env("GIGACHAT_VERIFY_SSL", True)
    if not verify_ssl:
        print(
            "WARNING: GIGACHAT_VERIFY_SSL=false is intended only for local debugging and is not recommended."
        )
        return False
    bundle_path = str(os.getenv("GIGACHAT_CA_BUNDLE", "") or "").strip()
    if bundle_path:
        bundle_file = Path(bundle_path)
        if not bundle_file.is_file():
            raise FileNotFoundError(f"GigaChat CA bundle file not found: {bundle_path}")
        return str(bundle_file)
    return True


def _get_gigachat_access_token(settings: Settings | None = None) -> str:
    """Return a cached GigaChat token or fetch a fresh one."""

    token, report = get_gigachat_access_token_with_status(settings)
    if not token:
        raise RuntimeError(report.get("error", "GigaChat token request failed."))
    return token


def get_gigachat_access_token_with_status(settings: Settings | None = None, *, force_refresh: bool = False) -> tuple[str, dict[str, object]]:
    """Fetch or reuse a cached GigaChat token with safe status reporting."""

    global _GIGACHAT_TOKEN_CACHE
    now = datetime.now(timezone.utc)
    if not force_refresh and _GIGACHAT_TOKEN_CACHE and _GIGACHAT_TOKEN_CACHE.expires_at > now:
        return _GIGACHAT_TOKEN_CACHE.access_token, {
            "token_status": _GIGACHAT_TOKEN_CACHE.token_status,
            "token_obtained": True,
            "token_requests_attempted": 0,
            "token_request_retries": 0,
            "error": "",
            "from_cache": True,
        }

    auth_key = (
        str(getattr(settings, "gigachat_auth_key", "") or "").strip()
        if settings is not None
        else os.getenv("GIGACHAT_AUTH_KEY", "").strip()
    )
    if not auth_key:
        return "", {
            "token_status": "missing_credentials",
            "token_obtained": False,
            "token_requests_attempted": 0,
            "token_request_retries": 0,
            "error": "GIGACHAT_AUTH_KEY is missing.",
            "from_cache": False,
        }

    scope = (
        str(getattr(settings, "gigachat_scope", DEFAULT_GIGACHAT_SCOPE) or "").strip()
        if settings is not None
        else os.getenv("GIGACHAT_SCOPE", DEFAULT_GIGACHAT_SCOPE).strip()
    ) or DEFAULT_GIGACHAT_SCOPE
    oauth_url = (
        str(getattr(settings, "gigachat_oauth_url", DEFAULT_GIGACHAT_OAUTH_URL) or "").strip()
        if settings is not None
        else os.getenv("GIGACHAT_OAUTH_URL", DEFAULT_GIGACHAT_OAUTH_URL).strip()
    ) or DEFAULT_GIGACHAT_OAUTH_URL
    verify_ssl = get_gigachat_verify_value(settings) if settings is not None else _resolve_gigachat_verify_without_settings()

    max_attempts = 3
    token_requests_attempted = 0
    token_request_retries = 0
    last_error = ""
    last_status = "token_error"

    for attempt in range(1, max_attempts + 1):
        token_requests_attempted += 1
        try:
            response = requests.post(
                oauth_url,
                headers={
                    "Content-Type": "application/x-www-form-urlencoded",
                    "Accept": "application/json",
                    "RqUID": str(uuid.uuid4()),
                    "Authorization": f"Basic {auth_key}",
                },
                data={"scope": scope},
                timeout=30,
                verify=verify_ssl,
            )
        except requests.exceptions.SSLError as exc:
            message = "GigaChat SSL verification failed. Install the required certificates or configure GIGACHAT_CA_BUNDLE."
            return "", {
                "token_status": "ssl_error",
                "token_obtained": False,
                "token_requests_attempted": token_requests_attempted,
                "token_request_retries": token_request_retries,
                "error": message,
                "from_cache": False,
            }
        except requests.exceptions.Timeout as exc:
            last_error = f"GigaChat token request timed out: {exc}"
            last_status = "timeout_error"
            if attempt < max_attempts:
                token_request_retries += 1
                sleep(1)
                continue
            return "", {
                "token_status": last_status,
                "token_obtained": False,
                "token_requests_attempted": token_requests_attempted,
                "token_request_retries": token_request_retries,
                "error": last_error,
                "from_cache": False,
            }
        except requests.exceptions.ConnectionError as exc:
            last_error = f"GigaChat token connection failed: {exc}"
            last_status = "connection_error"
            if attempt < max_attempts:
                token_request_retries += 1
                sleep(1)
                continue
            return "", {
                "token_status": last_status,
                "token_obtained": False,
                "token_requests_attempted": token_requests_attempted,
                "token_request_retries": token_request_retries,
                "error": last_error,
                "from_cache": False,
            }
        except requests.RequestException as exc:
            last_error = f"GigaChat token request failed: {exc}"
            last_status = "token_error"
            if attempt < max_attempts:
                token_request_retries += 1
                sleep(1)
                continue
            return "", {
                "token_status": last_status,
                "token_obtained": False,
                "token_requests_attempted": token_requests_attempted,
                "token_request_retries": token_request_retries,
                "error": last_error,
                "from_cache": False,
            }

        status_code = int(getattr(response, "status_code", 0) or 0)
        if status_code >= 500:
            last_status = "token_http_error"
            last_error = f"GigaChat token request failed with HTTP {status_code}."
            if attempt < max_attempts:
                token_request_retries += 1
                sleep(1)
                continue
            return "", {
                "token_status": last_status,
                "token_obtained": False,
                "token_requests_attempted": token_requests_attempted,
                "token_request_retries": token_request_retries,
                "error": last_error,
                "from_cache": False,
            }
        if status_code in {400, 401, 403}:
            last_status = "auth_error" if status_code in {401, 403} else "bad_request_error"
            last_error = f"GigaChat token request failed with HTTP {status_code}."
            return "", {
                "token_status": last_status,
                "token_obtained": False,
                "token_requests_attempted": token_requests_attempted,
                "token_request_retries": token_request_retries,
                "error": last_error,
                "from_cache": False,
            }
        try:
            response.raise_for_status()
        except requests.RequestException as exc:
            last_error = f"GigaChat token request failed: {exc}"
            last_status = "token_error"
            if attempt < max_attempts:
                token_request_retries += 1
                sleep(1)
                continue
            return "", {
                "token_status": last_status,
                "token_obtained": False,
                "token_requests_attempted": token_requests_attempted,
                "token_request_retries": token_request_retries,
                "error": last_error,
                "from_cache": False,
            }

        try:
            payload = response.json()
        except ValueError:
            last_error = "GigaChat token response was not valid JSON."
            last_status = "token_error"
            if attempt < max_attempts:
                token_request_retries += 1
                sleep(1)
                continue
            return "", {
                "token_status": last_status,
                "token_obtained": False,
                "token_requests_attempted": token_requests_attempted,
                "token_request_retries": token_request_retries,
                "error": last_error,
                "from_cache": False,
            }

        access_token = str(payload.get("access_token") or payload.get("token") or "").strip()
        if not access_token:
            last_error = "GigaChat token response did not contain an access token."
            last_status = "token_error"
            if attempt < max_attempts:
                token_request_retries += 1
                sleep(1)
                continue
            return "", {
                "token_status": last_status,
                "token_obtained": False,
                "token_requests_attempted": token_requests_attempted,
                "token_request_retries": token_request_retries,
                "error": last_error,
                "from_cache": False,
            }

        expires_at = _resolve_gigachat_expiration(payload, now)
        _GIGACHAT_TOKEN_CACHE = CachedToken(
            access_token=access_token,
            expires_at=expires_at,
            obtained_at=now,
            token_status="ok",
        )
        return access_token, {
            "token_status": "ok",
            "token_obtained": True,
            "token_requests_attempted": token_requests_attempted,
            "token_request_retries": token_request_retries,
            "error": "",
            "from_cache": False,
        }

    return "", {
        "token_status": last_status,
        "token_obtained": False,
        "token_requests_attempted": token_requests_attempted,
        "token_request_retries": token_request_retries,
        "error": last_error or "GigaChat token request failed.",
        "from_cache": False,
    }


def build_gigachat_token_preflight_report(
    settings: Settings | None = None,
    *,
    use_cache: bool = True,
) -> dict[str, object]:
    """Return a safe provider preflight report for GigaChat."""

    token, token_report = get_gigachat_access_token_with_status(settings, force_refresh=not use_cache)
    report = {
        "provider_preflight_ok": bool(token_report.get("token_obtained", False)),
        "provider_preflight_status": str(token_report.get("token_status", "token_error")),
        "final_token_status": str(token_report.get("token_status", "token_error")),
        "token_status": str(token_report.get("token_status", "token_error")),
        "token_obtained": bool(token_report.get("token_obtained", False)),
        "token_requests_attempted": int(token_report.get("token_requests_attempted", 0)),
        "token_request_retries": int(token_report.get("token_request_retries", 0)),
        "provider_preflight_error_type": str(token_report.get("token_status", "token_error")) if not token_report.get("token_obtained", False) else "",
        "provider_preflight_error_message_short": str(token_report.get("error", ""))[:180],
        "error": str(token_report.get("error", "")),
    }
    # Do not leak or persist tokens; this function only reports status.
    _ = token
    return report


def _resolve_gigachat_expiration(payload: dict[str, object], now: datetime) -> datetime:
    """Resolve token expiration from the GigaChat token payload."""

    expires_at_value = payload.get("expires_at") or payload.get("expiresAt")
    if isinstance(expires_at_value, (int, float)):
        if float(expires_at_value) > 10_000_000_000:
            return datetime.fromtimestamp(float(expires_at_value) / 1000.0, tz=timezone.utc)
        return datetime.fromtimestamp(float(expires_at_value), tz=timezone.utc)
    if isinstance(expires_at_value, str) and expires_at_value.strip():
        try:
            parsed = datetime.fromisoformat(expires_at_value.replace("Z", "+00:00"))
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
            return parsed.astimezone(timezone.utc)
        except ValueError:
            pass

    expires_in = payload.get("expires_in") or payload.get("expiresIn") or payload.get("expire")
    try:
        expires_seconds = float(expires_in)
    except (TypeError, ValueError):
        expires_seconds = float(DEFAULT_GIGACHAT_TOKEN_TTL_MINUTES * 60)
    expires_seconds = min(expires_seconds, float(DEFAULT_GIGACHAT_TOKEN_TTL_MINUTES * 60))
    return now + timedelta(seconds=expires_seconds)


def _parse_bool_env(name: str, default: bool) -> bool:
    """Parse a boolean environment value."""

    raw = os.getenv(name)
    if raw is None:
        return default
    normalized = raw.strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    return default
