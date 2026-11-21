"""HTTP transport pinned to validated addresses."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Callable, Mapping, Protocol
from urllib.parse import urljoin, urlparse
import http.client
import ipaddress
import socket
import ssl

from ..literature_text import AcquisitionError


class SafeHTTPError(AcquisitionError):
    def __init__(self, code: str, message: str, *, status: int | None = None) -> None:
        self.code = code
        self.status = status
        super().__init__(f"{code}: {message}")


@dataclass(frozen=True)
class HTTPResponse:
    url: str
    status: int
    headers: Mapping[str, str]
    body: bytes
    connected_ip: str


class HTTPTransport(Protocol):
    def request(
        self,
        url: str,
        *,
        validated_ips: tuple[str, ...],
        timeout: float,
        max_bytes: int,
    ) -> HTTPResponse:
        """Connect only to one of validated_ips; never resolve or follow redirects."""
        ...


class _PinnedHTTPConnection(http.client.HTTPConnection):
    def __init__(self, hostname: str, port: int, pinned_ip: str, timeout: float) -> None:
        super().__init__(hostname, port, timeout=timeout)
        self.pinned_ip = pinned_ip

    def connect(self) -> None:
        self.sock = socket.create_connection(
            (self.pinned_ip, self.port), self.timeout, self.source_address
        )


class _PinnedHTTPSConnection(_PinnedHTTPConnection):
    def __init__(
        self,
        hostname: str,
        port: int,
        pinned_ip: str,
        timeout: float,
        context: ssl.SSLContext,
    ) -> None:
        super().__init__(hostname, port, pinned_ip, timeout)
        self.context = context

    def connect(self) -> None:
        super().connect()
        if self.sock is None:
            raise OSError("pinned HTTPS socket was not created")
        self.sock = self.context.wrap_socket(self.sock, server_hostname=self.host)


class PinnedHTTPTransport:
    """HTTP transport that connects to a prevalidated IP and preserves Host/TLS SNI."""

    def __init__(self, *, ssl_context: ssl.SSLContext | None = None) -> None:
        self.ssl_context = ssl_context or ssl.create_default_context()

    def request(
        self,
        url: str,
        *,
        validated_ips: tuple[str, ...],
        timeout: float,
        max_bytes: int,
    ) -> HTTPResponse:
        if not validated_ips:
            raise SafeHTTPError("DNS_RESOLUTION_FAILED", "no validated IP was supplied")
        parsed = urlparse(url)
        hostname = parsed.hostname or ""
        try:
            port = parsed.port or (443 if parsed.scheme.casefold() == "https" else 80)
        except ValueError as error:
            raise SafeHTTPError("INVALID_URL", "URL port is invalid") from error
        target = parsed.path or "/"
        if parsed.query:
            target = f"{target}?{parsed.query}"
        host_header = f"[{hostname}]" if ":" in hostname else hostname
        default_port = 443 if parsed.scheme.casefold() == "https" else 80
        if port != default_port:
            host_header = f"{hostname}:{port}"
        last_error: OSError | None = None
        for pinned_ip in validated_ips:
            connection: http.client.HTTPConnection
            if parsed.scheme.casefold() == "https":
                connection = _PinnedHTTPSConnection(
                    hostname, port, pinned_ip, timeout, self.ssl_context
                )
            else:
                connection = _PinnedHTTPConnection(
                    hostname, port, pinned_ip, timeout
                )
            try:
                connection.request(
                    "GET",
                    target,
                    headers={
                        "Host": host_header,
                        "User-Agent": "SPC-Literature-Acquisition/1.1",
                        "Accept": "*/*",
                        "Connection": "close",
                    },
                )
                response = connection.getresponse()
                body = response.read(max_bytes + 1)
                headers = MappingProxyType(
                    {
                        key.casefold(): value.strip()
                        for key, value in response.getheaders()
                    }
                )
                return HTTPResponse(
                    url=url,
                    status=response.status,
                    headers=headers,
                    body=body,
                    connected_ip=pinned_ip,
                )
            except OSError as error:
                last_error = error
            finally:
                connection.close()
        raise SafeHTTPError(
            "NETWORK_ERROR", "remote resource could not be fetched"
        ) from last_error


class SafeHTTPFetcher:
    def __init__(
        self,
        transport: HTTPTransport | None = None,
        *,
        timeout: float = 15.0,
        max_bytes: int = 20 * 1024 * 1024,
        max_redirects: int = 5,
        dns_resolver: Callable[[str], tuple[str, ...]] | None = None,
    ) -> None:
        if timeout <= 0 or max_bytes <= 0 or max_redirects < 0:
            raise ValueError("safe HTTP limits must be positive")
        self.transport = transport or PinnedHTTPTransport()
        self.timeout = timeout
        self.max_bytes = max_bytes
        self.max_redirects = max_redirects
        self.dns_resolver = dns_resolver or self._resolve_host

    def fetch(
        self, url: str, *, allowed_content_types: frozenset[str]
    ) -> HTTPResponse:
        current = url
        for redirect_count in range(self.max_redirects + 1):
            validated_ips = self._validated_ips(current)
            response = self.transport.request(
                current,
                validated_ips=validated_ips,
                timeout=self.timeout,
                max_bytes=self.max_bytes,
            )
            if response.connected_ip not in validated_ips:
                raise SafeHTTPError(
                    "CONNECTION_IP_MISMATCH",
                    "transport connected outside the validated IP set",
                )
            if response.url != current:
                raise SafeHTTPError(
                    "UNVALIDATED_REDIRECT",
                    "HTTP transport followed a redirect outside the safe fetcher",
                )
            headers = {
                str(key).casefold(): str(value).strip()
                for key, value in response.headers.items()
            }
            if response.status in {301, 302, 303, 307, 308}:
                location = headers.get("location")
                if not location:
                    raise SafeHTTPError("INVALID_REDIRECT", "redirect has no location")
                if redirect_count == self.max_redirects:
                    raise SafeHTTPError("TOO_MANY_REDIRECTS", "redirect limit exceeded")
                current = urljoin(current, location)
                continue
            if response.status in {401, 403}:
                raise SafeHTTPError(
                    "REQUIRES_AUTHENTICATION",
                    "remote resource requires authentication",
                    status=response.status,
                )
            if response.status < 200 or response.status >= 300:
                raise SafeHTTPError(
                    "HTTP_STATUS_ERROR",
                    f"remote resource returned HTTP {response.status}",
                    status=response.status,
                )
            declared_length = headers.get("content-length")
            if declared_length is not None:
                try:
                    if int(declared_length) > self.max_bytes:
                        raise SafeHTTPError(
                            "RESOURCE_TOO_LARGE", "declared resource size exceeds limit"
                        )
                except ValueError as error:
                    raise SafeHTTPError(
                        "INVALID_CONTENT_LENGTH", "invalid Content-Length header"
                    ) from error
            if len(response.body) > self.max_bytes:
                raise SafeHTTPError(
                    "RESOURCE_TOO_LARGE", "downloaded resource exceeds limit"
                )
            media_type = headers.get("content-type", "").split(";", 1)[0]
            media_type = media_type.strip().casefold()
            if media_type not in allowed_content_types:
                raise SafeHTTPError(
                    "UNSUPPORTED_CONTENT_TYPE",
                    f"content type is not allowed: {media_type or 'missing'}",
                )
            return HTTPResponse(
                url=current,
                status=response.status,
                headers=MappingProxyType(headers),
                body=response.body,
                connected_ip=response.connected_ip,
            )
        raise SafeHTTPError("TOO_MANY_REDIRECTS", "redirect limit exceeded")

    def _validated_ips(self, url: str) -> tuple[str, ...]:
        parsed = urlparse(url)
        if parsed.scheme.casefold() not in {"http", "https"}:
            raise SafeHTTPError("UNSAFE_URL_SCHEME", "only HTTP/HTTPS are allowed")
        if parsed.username is not None or parsed.password is not None:
            raise SafeHTTPError("URL_CREDENTIALS_REJECTED", "URL credentials are forbidden")
        hostname = parsed.hostname
        if not hostname:
            raise SafeHTTPError("INVALID_URL", "URL has no hostname")
        normalized_host = hostname.rstrip(".").casefold()
        if normalized_host == "localhost" or normalized_host.endswith(".localhost"):
            raise SafeHTTPError("LOCALHOST_REJECTED", "localhost is forbidden")
        try:
            addresses = (str(ipaddress.ip_address(normalized_host)),)
        except ValueError:
            try:
                addresses = self.dns_resolver(normalized_host)
            except OSError as error:
                raise SafeHTTPError("DNS_RESOLUTION_FAILED", "hostname could not resolve") from error
        if not addresses:
            raise SafeHTTPError("DNS_RESOLUTION_FAILED", "hostname resolved to no addresses")
        validated: list[str] = []
        for address in addresses:
            try:
                parsed_address = ipaddress.ip_address(address)
            except ValueError as error:
                raise SafeHTTPError("INVALID_DNS_ADDRESS", "DNS returned an invalid address") from error
            if not parsed_address.is_global:
                raise SafeHTTPError(
                    "PRIVATE_ADDRESS_REJECTED",
                    "loopback, private, link-local, or non-global address is forbidden",
                )
            validated.append(str(parsed_address))
        return tuple(sorted(set(validated)))

    @staticmethod
    def _resolve_host(hostname: str) -> tuple[str, ...]:
        return tuple(
            sorted(
                {
                    item[4][0]
                    for item in socket.getaddrinfo(
                        hostname, None, type=socket.SOCK_STREAM
                    )
                }
            )
        )
