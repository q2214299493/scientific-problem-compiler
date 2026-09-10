from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
from typing import Any

from ...serialization import content_hash
from .contracts import LiteratureKnowledgeProviderInvocation


CODEX_TRANSPORT_ID = "codex-cli"
CODEX_TRANSPORT_VERSION = "1.1.0"
DEFAULT_TIMEOUT_SECONDS = 300.0
DEFAULT_MAX_INPUT_BYTES = 1_000_000
DEFAULT_MAX_OUTPUT_BYTES = 1_000_000
PROBE_MAX_OUTPUT_BYTES = 65_536
CODEX_DISABLED_FEATURES = (
    "shell_tool",
    "shell_snapshot",
    "standalone_web_search",
)
CODEX_FEATURE_OVERRIDES = tuple(f"features.{feature}=false" for feature in CODEX_DISABLED_FEATURES)
# Cross-platform process/bootstrap, locale, TLS, and Codex state/auth discovery
# only. Provider, cloud, database, and credential-like variables are absent.
# CODEX_ACCESS_TOKEN is intentionally excluded: this transport requires the
# existing local ChatGPT login managed by Codex itself.
CODEX_ENVIRONMENT_ALLOWLIST = (
    "PATH",
    "HOME",
    "USER",
    "LOGNAME",
    "USERPROFILE",
    "HOMEDRIVE",
    "HOMEPATH",
    "APPDATA",
    "LOCALAPPDATA",
    "SYSTEMROOT",
    "WINDIR",
    "COMSPEC",
    "PATHEXT",
    "TEMP",
    "TMP",
    "TMPDIR",
    "XDG_CONFIG_HOME",
    "XDG_DATA_HOME",
    "XDG_CACHE_HOME",
    "XDG_RUNTIME_DIR",
    "LANG",
    "LC_ALL",
    "LC_CTYPE",
    "SSL_CERT_FILE",
    "SSL_CERT_DIR",
    "CODEX_HOME",
)
ALLOWED_MODEL_ITEM_TYPES = frozenset({"agent_message", "reasoning"})
DISALLOWED_EVENT_MARKERS = frozenset({"command", "tool", "file", "web", "mcp"})


class CodexCLIUnavailableError(RuntimeError):
    pass


class CodexCLIExecutionError(RuntimeError):
    pass


@dataclass(frozen=True)
class CodexCLIRuntime:
    executable: str
    cli_version: str
    selected_model: str
    authenticated: bool
    authentication_status: str
    disabled_features: tuple[str, ...]


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _bounded_bytes(path: Path, limit: int, *, label: str) -> bytes:
    if path.stat().st_size > limit:
        raise CodexCLIExecutionError(f"Codex CLI {label} exceeded {limit} bytes")
    return path.read_bytes()


def _minimal_codex_environment(source: Mapping[str, str]) -> dict[str, str]:
    return {
        variable: source[variable]
        for variable in CODEX_ENVIRONMENT_ALLOWLIST
        if variable in source and source[variable]
    }


def _feature_override_arguments() -> list[str]:
    return [part for override in CODEX_FEATURE_OVERRIDES for part in ("-c", override)]


class CodexCLILLMTransport:
    """Structured K1F transport backed by an authenticated local Codex CLI."""

    def __init__(
        self,
        executable: str | os.PathLike[str] | Sequence[str] = "codex",
        *,
        model: str,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
        max_output_bytes: int = DEFAULT_MAX_OUTPUT_BYTES,
        environment: Mapping[str, str] | None = None,
    ) -> None:
        if isinstance(executable, (str, os.PathLike)):
            command = (os.fspath(executable),)
        else:
            command = tuple(executable)
        if not command or any(not str(part).strip() for part in command):
            raise ValueError("Codex CLI executable command must not be blank")
        if not model.strip():
            raise ValueError("an explicit Codex CLI model is required")
        if not math.isfinite(timeout_seconds) or not 1 <= timeout_seconds <= 1800:
            raise ValueError("Codex CLI timeout must be between 1 and 1800 seconds")
        if not 1 <= max_output_bytes <= DEFAULT_MAX_OUTPUT_BYTES:
            raise ValueError(f"Codex CLI output limit must be between 1 and {DEFAULT_MAX_OUTPUT_BYTES} bytes")
        self.command = tuple(str(part) for part in command)
        self.selected_model = model.strip()
        self.model_id = self.selected_model
        self.timeout_seconds = timeout_seconds
        self.max_output_bytes = max_output_bytes
        environment_source = environment if environment is not None else os.environ
        self._environment = _minimal_codex_environment(environment_source)
        self._runtime: CodexCLIRuntime | None = None
        self.last_invocation: LiteratureKnowledgeProviderInvocation | None = None

    def _probe(
        self,
        arguments: Sequence[str],
    ) -> subprocess.CompletedProcess[bytes]:
        try:
            completed = subprocess.run(
                [*self.command, *arguments],
                input=b"",
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=15,
                check=False,
                shell=False,
                env=self._environment,
            )
        except FileNotFoundError as error:
            raise CodexCLIUnavailableError(
                "Codex CLI is not installed or is not on PATH. Install Codex CLI, "
                "then run 'codex login' and complete the ChatGPT sign-in flow."
            ) from error
        except subprocess.TimeoutExpired as error:
            raise CodexCLIUnavailableError("Codex CLI availability check timed out") from error
        if len(completed.stdout) > PROBE_MAX_OUTPUT_BYTES or len(completed.stderr) > PROBE_MAX_OUTPUT_BYTES:
            raise CodexCLIUnavailableError("Codex CLI availability check produced oversized output")
        return completed

    def _execution_arguments(
        self,
        temporary_root: Path,
        schema_path: Path,
        output_path: Path,
        *,
        help_only: bool = False,
    ) -> list[str]:
        arguments = [
            "exec",
            *_feature_override_arguments(),
            "--strict-config",
            "-c",
            'approval_policy="never"',
            "--sandbox",
            "read-only",
            "--ephemeral",
            "--ignore-user-config",
            "--ignore-rules",
            "--skip-git-repo-check",
            "--json",
            "--color",
            "never",
            "--cd",
            str(temporary_root),
            "--output-schema",
            str(schema_path),
            "--output-last-message",
            str(output_path),
            "--model",
            self.selected_model,
        ]
        arguments.append("--help" if help_only else "-")
        return arguments

    def _verify_configuration(self) -> None:
        feature_check = self._probe((*_feature_override_arguments(), "features", "list"))
        if feature_check.returncode != 0:
            raise CodexCLIUnavailableError("Codex CLI rejected the required tool-disable configuration")
        effective: dict[str, str] = {}
        for line in feature_check.stdout.decode(
            "utf-8",
            errors="replace",
        ).splitlines():
            parts = line.split()
            if len(parts) >= 3 and parts[0] in CODEX_DISABLED_FEATURES:
                effective[parts[0]] = parts[-1].casefold()
        invalid = tuple(feature for feature in CODEX_DISABLED_FEATURES if effective.get(feature) != "false")
        if invalid:
            raise CodexCLIUnavailableError("Codex CLI did not accept required disabled features: " + ", ".join(invalid))
        with tempfile.TemporaryDirectory(prefix="spc-k1f-codex-preflight-") as directory:
            temporary_root = Path(directory)
            schema_path = temporary_root / "response-schema.json"
            output_path = temporary_root / "response.json"
            schema_path.write_text("{}", encoding="utf-8")
            help_check = self._probe(
                self._execution_arguments(
                    temporary_root,
                    schema_path,
                    output_path,
                    help_only=True,
                )
            )
        if help_check.returncode != 0:
            raise CodexCLIUnavailableError("Codex CLI rejected required non-interactive extraction flags")

    def inspect_runtime(self, *, refresh: bool = False) -> CodexCLIRuntime:
        if self._runtime is not None and not refresh:
            return self._runtime
        if len(self.command) == 1:
            executable = self.command[0]
            if (
                not Path(executable).is_file()
                and shutil.which(
                    executable,
                    path=self._environment.get("PATH"),
                )
                is None
            ):
                raise CodexCLIUnavailableError(
                    "Codex CLI is not installed or is not on PATH. Install Codex "
                    "CLI, then run 'codex login' and complete the ChatGPT sign-in "
                    "flow."
                )
        version = self._probe(("--version",))
        if version.returncode != 0:
            raise CodexCLIUnavailableError("Codex CLI version check failed")
        version_text = version.stdout.decode(
            "utf-8",
            errors="replace",
        ).strip()
        if not version_text:
            raise CodexCLIUnavailableError("Codex CLI did not report a version")
        login = self._probe(("login", "status"))
        status_text = (
            b"\n".join((login.stdout, login.stderr))
            .decode(
                "utf-8",
                errors="replace",
            )
            .strip()
        )
        authenticated = login.returncode == 0 and "chatgpt" in status_text.casefold()
        if not authenticated:
            raise CodexCLIUnavailableError(
                "Codex CLI is not authenticated with ChatGPT. Run 'codex login', "
                "complete the ChatGPT sign-in flow, and verify it with 'codex "
                "login status'. API-key login is not accepted by this K1F "
                "transport."
            )
        self._verify_configuration()
        self._runtime = CodexCLIRuntime(
            executable=Path(self.command[0]).name,
            cli_version=version_text,
            selected_model=self.selected_model,
            authenticated=True,
            authentication_status="chatgpt",
            disabled_features=CODEX_DISABLED_FEATURES,
        )
        return self._runtime

    @staticmethod
    def _audit_event_stream(raw_events: bytes) -> None:
        for line_number, raw_line in enumerate(
            raw_events.splitlines(),
            start=1,
        ):
            if not raw_line.strip():
                continue
            try:
                event = json.loads(raw_line)
            except json.JSONDecodeError as error:
                raise CodexCLIExecutionError(f"Codex CLI emitted invalid JSONL at event line {line_number}") from error
            if not isinstance(event, dict):
                raise CodexCLIExecutionError("Codex CLI event stream must contain JSON objects")
            event_type = str(event.get("type", "")).casefold()
            event_activity = event_type.removeprefix("item.")
            if any(marker in event_activity for marker in DISALLOWED_EVENT_MARKERS):
                raise CodexCLIExecutionError(f"Codex CLI attempted a disallowed extraction event: {event_type}")
            item = event.get("item")
            item_type = item.get("type") if isinstance(item, dict) else None
            if item_type is not None and item_type not in ALLOWED_MODEL_ITEM_TYPES:
                raise CodexCLIExecutionError(f"Codex CLI attempted a disallowed extraction action: {item_type}")

    def generate_structured(
        self,
        *,
        system_prompt: str,
        input_payload: dict[str, Any],
        response_schema: dict[str, Any],
        temperature: float,
    ) -> str:
        if temperature != 0.0:
            raise ValueError("Codex CLI K1F extraction requires temperature=0")
        runtime = self.inspect_runtime()
        prompt_payload = {
            "instructions": system_prompt,
            "scientific_input": input_payload,
        }
        prompt = (
            "Return exactly one JSON object matching the supplied output schema. "
            "Use only the bounded scientific input below.\n"
            + json.dumps(
                prompt_payload,
                sort_keys=True,
                ensure_ascii=False,
                separators=(",", ":"),
            )
        ).encode("utf-8")
        if len(prompt) > DEFAULT_MAX_INPUT_BYTES:
            raise ValueError(f"Codex CLI scientific input exceeded {DEFAULT_MAX_INPUT_BYTES} bytes")
        schema_bytes = json.dumps(
            response_schema,
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
        input_hash = _sha256_bytes(prompt)

        with tempfile.TemporaryDirectory(prefix="spc-k1f-codex-") as temporary_directory:
            temporary_root = Path(temporary_directory)
            schema_path = temporary_root / "response-schema.json"
            output_path = temporary_root / "response.json"
            stdout_path = temporary_root / "events.jsonl"
            stderr_path = temporary_root / "stderr.txt"
            schema_path.write_bytes(schema_bytes)
            arguments = self._execution_arguments(
                temporary_root,
                schema_path,
                output_path,
            )
            try:
                with stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
                    completed = subprocess.run(
                        [*self.command, *arguments],
                        input=prompt,
                        stdout=stdout,
                        stderr=stderr,
                        timeout=self.timeout_seconds,
                        check=False,
                        shell=False,
                        env=self._environment,
                    )
            except subprocess.TimeoutExpired as error:
                raise CodexCLIExecutionError(
                    f"Codex CLI extraction timed out after {self.timeout_seconds:g} seconds"
                ) from error
            events = _bounded_bytes(
                stdout_path,
                self.max_output_bytes,
                label="event output",
            )
            stderr = _bounded_bytes(
                stderr_path,
                PROBE_MAX_OUTPUT_BYTES,
                label="diagnostic output",
            )
            if completed.returncode != 0:
                diagnostic = stderr.decode(
                    "utf-8",
                    errors="replace",
                ).strip()
                raise CodexCLIExecutionError(
                    "Codex CLI extraction failed with exit code "
                    f"{completed.returncode}" + (f": {diagnostic}" if diagnostic else "")
                )
            self._audit_event_stream(events)
            if not output_path.is_file() or output_path.is_symlink():
                raise CodexCLIExecutionError("Codex CLI did not create a regular structured output")
            output = _bounded_bytes(
                output_path,
                self.max_output_bytes,
                label="structured output",
            )
        try:
            decoded_output = output.decode("utf-8")
        except UnicodeDecodeError as error:
            raise CodexCLIExecutionError("Codex CLI output is not valid UTF-8") from error

        invocation_config = {
            "transport_id": CODEX_TRANSPORT_ID,
            "transport_version": CODEX_TRANSPORT_VERSION,
            "runtime_version": runtime.cli_version,
            "selected_model": self.selected_model,
            "feature_overrides": CODEX_FEATURE_OVERRIDES,
            "shell_tool": False,
            "shell_snapshot": False,
            "standalone_web_search": False,
            "strict_config": True,
            "sandbox": "read-only",
            "approval_policy": "never",
            "ephemeral": True,
            "ignore_user_config": True,
            "ignore_rules": True,
            "web_search": False,
            "environment_allowlist": CODEX_ENVIRONMENT_ALLOWLIST,
            "authentication_mode": "chatgpt-local-session",
            "working_directory": "isolated-temporary-directory",
            "timeout_seconds": self.timeout_seconds,
            "max_input_bytes": DEFAULT_MAX_INPUT_BYTES,
            "max_output_bytes": self.max_output_bytes,
            "response_schema_hash": _sha256_bytes(schema_bytes),
        }
        invocation_config_hash = content_hash(invocation_config)
        identity = {
            "transport_id": CODEX_TRANSPORT_ID,
            "transport_version": CODEX_TRANSPORT_VERSION,
            "runtime_version": runtime.cli_version,
            "selected_model": self.selected_model,
            "invocation_config_hash": invocation_config_hash,
            "input_hash": input_hash,
            "output_hash": _sha256_bytes(output),
        }
        invocation_id = f"literature-knowledge-provider-invocation-{content_hash(identity)[:24]}"
        payload = {"invocation_id": invocation_id, **identity}
        self.last_invocation = LiteratureKnowledgeProviderInvocation(
            **payload,
            content_hash=content_hash(payload),
        )
        return decoded_output


__all__ = [
    "CODEX_DISABLED_FEATURES",
    "CODEX_ENVIRONMENT_ALLOWLIST",
    "CODEX_FEATURE_OVERRIDES",
    "CODEX_TRANSPORT_ID",
    "CODEX_TRANSPORT_VERSION",
    "CodexCLIExecutionError",
    "CodexCLILLMTransport",
    "CodexCLIRuntime",
    "CodexCLIUnavailableError",
]
