"""Build a Postman Collection v2.1 from scanned endpoints, and the tool that writes it to disk."""

from __future__ import annotations

import asyncio
import json
import re
import uuid
from collections import Counter
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from app.tools.api_scanner import Endpoint, ScanResult, scan_project
from app.tools.base import ToolResult
from app.tools.filesystem import AccessDenied, FileAccessPolicy, FileTool

SCHEMA = "https://schema.getpostman.com/json/collection/v2.1.0/collection.json"
_LOGIN_PATH = re.compile(r"(?:create|authenticate)user|login|verifyotp|signin|token", re.I)
_TOKEN_SCRIPT = [
    "// Saves the auth token from the response so other requests can use {{token}}.",
    "let j = {};",
    "try { j = pm.response.json(); } catch (e) {}",
    "const d = j.data || {};",
    "const token = j.token || d.token || (j.user && j.user.token) || (d.user && d.user.token)",
    "  || j.accessToken || j.access_token || d.accessToken || d.access_token;",
    "if (token) { pm.collectionVariables.set('token', token); }",
]


def slugify(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "-", name).strip("-").lower() or "api"


def _url(path: str, query: dict[str, str]) -> dict[str, Any]:
    if path.startswith(("http://", "https://")):
        parsed = urlparse(path)
        url: dict[str, Any] = {
            "raw": path, "protocol": parsed.scheme, "host": parsed.netloc.split("."),
            "path": [s for s in parsed.path.split("/") if s],
        }
        return url
    segments = [s for s in path.strip("/").split("/") if s]
    segments = [f":{s[1:-1]}" if s.startswith("{") and s.endswith("}") else s for s in segments]
    raw = "{{baseUrl}}/" + "/".join(segments)
    url: dict[str, Any] = {"raw": raw, "host": ["{{baseUrl}}"], "path": segments}
    if query:
        url["query"] = [{"key": k, "value": v} for k, v in query.items()]
        url["raw"] += "?" + "&".join(f"{k}={v}" for k, v in query.items())
    return url


def _description(ep: Endpoint) -> str:
    lines = []
    if ep.description:
        lines.append(ep.description)
    meta = []
    if ep.status:
        meta.append(f"Status: {ep.status}")
    if ep.auth is not None:
        meta.append("Auth: Bearer token required" if ep.auth else "Auth: none")
    if ep.body_type:
        meta.append(f"Body type: {ep.body_type}")
    if meta:
        lines.append(" | ".join(meta))
    if ep.body_type.startswith(("Map", "HashMap", "MutableMap")):
        lines.append("Body is a key/value map: only the keys visible in the code are listed; the app may send more.")
    if ep.body is None and ep.form is None and ep.method in ("POST", "PUT", "PATCH"):
        lines.append("Request body is unknown (not found in the code): fill it in.")
    lines.extend(ep.notes)
    lines.append("Source: " + "; ".join(ep.sources[:3]))
    return "\n\n".join(lines)


_FRAMEWORK = ("fastapi", "flask", "spring", "openapi")


def humanize(name: str) -> str:
    words = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", name).replace("_", " ").strip().lower()
    return words[:1].upper() + words[1:]


def display_name(ep: Endpoint) -> str:
    """Readable request name. Framework routes use their handler/operation name ("Create product"),
    plain Express routes use "METHOD /path", RPC-style APIs use the last URL segment (createUser)."""
    if ep.name and any(src.startswith(_FRAMEWORK) for src in ep.sources):
        return humanize(ep.name)
    if ep.sources and all(src.startswith("express") for src in ep.sources):
        return f"{ep.method} /{ep.path.strip('/')}"
    parts = [seg for seg in (urlparse(ep.path).path if ep.path.startswith("http") else ep.path).split("/")
             if seg and not seg.startswith(("{", ":"))]
    return parts[-1] if parts else (ep.name or ep.method)


def request_item(ep: Endpoint) -> dict[str, Any]:
    name = display_name(ep)
    if ep.variant:
        name = f"{name} [{ep.variant}]" if ep.variant not in name else name
    request: dict[str, Any] = {
        "method": ep.method,
        "header": [],
        "url": _url(ep.path, ep.query),
        "description": _description(ep),
    }
    if ep.auth is False:
        request["auth"] = {"type": "noauth"}
    if ep.form is not None:
        request["body"] = {
            "mode": "formdata",
            "formdata": [
                {"key": k, "type": "file", "src": []} if kind == "file" else {"key": k, "value": ex, "type": "text"}
                for k, kind, ex in ep.form
            ],
        }
    elif ep.method in ("POST", "PUT", "PATCH", "DELETE") and (ep.body is not None or ep.method != "DELETE"):
        request["header"].append({"key": "Content-Type", "value": "application/json"})
        request["body"] = {
            "mode": "raw",
            "raw": json.dumps(ep.body if ep.body is not None else {}, indent=2, ensure_ascii=False),
            "options": {"raw": {"language": "json"}},
        }
    item: dict[str, Any] = {"name": name, "request": request, "response": []}
    if _LOGIN_PATH.search(ep.path) and ep.method == "POST":
        item["event"] = [{"listen": "test", "script": {"type": "text/javascript", "exec": _TOKEN_SCRIPT}}]
    return item


def build_collection(name: str, endpoints: list[Endpoint], base_url: str) -> dict[str, Any]:
    folders: dict[str, list[dict[str, Any]]] = {}
    used_names: dict[str, Counter[str]] = {}
    for ep in endpoints:
        item = request_item(ep)
        seen = used_names.setdefault(ep.folder, Counter())
        seen[item["name"]] += 1
        if seen[item["name"]] > 1:
            item["name"] = f"{item['name']} ({ep.method})" if seen[item["name"]] == 2 else f"{item['name']} ({seen[item['name']]})"
        folders.setdefault(ep.folder, []).append(item)
    return {
        "info": {
            "_postman_id": str(uuid.uuid4()),
            "name": name,
            "description": "Generated by Zira from the project's code and docs. Review request bodies before relying on them.",
            "schema": SCHEMA,
        },
        "auth": {"type": "bearer", "bearer": [{"key": "token", "value": "{{token}}", "type": "string"}]},
        "variable": [
            {"key": "baseUrl", "value": base_url},
            {"key": "token", "value": ""},
        ],
        "item": [{"name": folder, "item": items} for folder, items in sorted(folders.items())],
    }


def summarise(result: ScanResult, collection: dict[str, Any]) -> str:
    eps = result.endpoints
    folders = Counter(e.folder for e in eps)
    with_body = sum(1 for e in eps if e.from_code and (e.body not in (None, {}) or e.form))
    docs_only = sum(1 for e in eps if not e.from_code)
    lines = [
        f'Postman collection "{collection["info"]["name"]}": {len(eps)} requests in {len(folders)} folders.',
        "Folders: " + ", ".join(f"{f} ({n})" for f, n in folders.most_common(14)),
        f"Bodies: {with_body} filled in from the code; {docs_only} endpoints known only from docs (empty placeholder bodies).",
        "Found via: " + (", ".join(result.stacks) or "nothing"),
        f"Base URL: {result.base_url} (collection variable baseUrl). "
        + (
            "Auth: Bearer {{token}}, saved automatically by the login requests."
            if any(_LOGIN_PATH.search(e.path) and e.method == "POST" for e in eps)
            else "Auth: the collection uses Bearer {{token}}; no login endpoint was found, so set the token variable yourself."
        ),
    ]
    lines += [f"Warning: {w}" for w in result.warnings]
    lines.append("Import in Postman with File > Import.")
    return "\n".join(lines)


class GeneratePostmanCollectionTool(FileTool):
    modes = frozenset({"chat", "edit"})  # it writes a file (to the exports folder), so not in read-only Plan mode
    name = "generate_postman_collection"
    description = (
        "Scan a project folder for its HTTP API (Retrofit, Express, FastAPI/Flask, Spring, OpenAPI, API docs) "
        "and write a Postman collection file the user can import. Returns a summary and a download link."
    )
    parameters = {
        "type": "object",
        "properties": {
            "project_path": {"type": "string", "description": "Project folder to scan"},
            "name": {"type": "string", "description": "Collection name (optional)"},
        },
        "required": ["project_path"],
    }

    def __init__(self, policy: FileAccessPolicy, exports_dir: Path, public_url: str) -> None:
        super().__init__(policy)
        self._exports = exports_dir
        self._public = public_url.rstrip("/")

    def describe(self, arguments: dict[str, Any]) -> str:
        return f"Scanning {self._rel(arguments, 'project_path')} for its API"

    async def execute(self, **arguments: Any) -> ToolResult:
        raw = arguments.get("project_path") or arguments.get("path")
        try:
            base = self._policy.resolve(raw if isinstance(raw, str) else "")
            if not base.is_dir():
                raise AccessDenied(f"'{self._policy.display(base)}' is not a directory.")
        except AccessDenied as exc:
            return ToolResult.failure(str(exc))
        name = arguments.get("name") if isinstance(arguments.get("name"), str) and arguments["name"].strip() else f"{base.name} API"
        result = await asyncio.to_thread(scan_project, self._policy, base)
        if not result.endpoints:
            return ToolResult.failure("No API endpoints were found. " + " ".join(result.warnings))
        collection = build_collection(name.strip()[:80], result.endpoints, result.base_url)
        filename = f"{slugify(name)}.postman_collection.json"
        self._exports.mkdir(parents=True, exist_ok=True)
        target = self._exports / filename
        target.write_text(json.dumps(collection, indent=2, ensure_ascii=False), encoding="utf-8")
        url = f"{self._public}/api/exports/{filename}"
        return ToolResult.success(
            summarise(result, collection) + f"\nSaved as {target}",
            files=[{"title": filename, "url": url}],
        )
