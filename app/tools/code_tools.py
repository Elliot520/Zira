"""Tools that help the model understand an unfamiliar codebase: a project overview and code outlines."""

from __future__ import annotations

import asyncio
import json
import os
import re
from collections import Counter
from pathlib import Path
from typing import Any

from app.tools.api_scanner import scan_project
from app.tools.base import ToolResult
from app.tools.filesystem import (
    SKIP_DIR_NAMES,
    UNTRUSTED_NOTE,
    AccessDenied,
    FileAccessPolicy,
    FileTool,
    redact,
)

STACK_MARKERS = {
    "build.gradle": "Gradle", "build.gradle.kts": "Gradle (Kotlin DSL)", "AndroidManifest.xml": "Android app",
    "package.json": "Node.js / JavaScript", "requirements.txt": "Python", "pyproject.toml": "Python",
    "setup.py": "Python", "pom.xml": "Maven / Java", "go.mod": "Go", "Cargo.toml": "Rust",
    "composer.json": "PHP", "Gemfile": "Ruby", "pubspec.yaml": "Dart / Flutter", "Package.swift": "Swift",
    "manage.py": "Django",
}
API_HINTS = {
    "Retrofit endpoints": re.compile(r"@(?:GET|POST|PUT|DELETE|PATCH)\b"),
    "Express routes": re.compile(r"\b(?:router|app)\.(?:get|post|put|delete|patch)\(\s*['\"`]/?"),
    "FastAPI/Flask routes": re.compile(r"@\w+\.(?:get|post|put|delete|patch|route)\(\s*['\"]"),
    "Spring mappings": re.compile(r"@(?:Get|Post|Put|Delete|Patch|Request)Mapping\b"),
}
SOURCE_EXTS = {".kt", ".java", ".py", ".js", ".ts", ".jsx", ".tsx", ".go", ".rs", ".php", ".rb", ".swift", ".cs", ".dart", ".scala"}
MAX_SCAN_BYTES = 300_000
MAX_OVERVIEW_CHARS = 5200
MAX_ENDPOINT_LINES = 30


def _read_small(policy: FileAccessPolicy, path: Path) -> str:
    try:
        return policy.read_text(path, max_bytes=MAX_SCAN_BYTES)
    except AccessDenied:
        return ""


class ProjectOverviewTool(FileTool):
    name = "project_overview"
    description = (
        "Summarise a project folder: tech stack, folder layout, file types, documentation and API-related "
        "files. Start with this when asked to understand or work on a project."
    )
    parameters = {
        "type": "object",
        "properties": {"path": {"type": "string", "description": "Project folder"}},
        "required": ["path"],
    }

    def describe(self, arguments: dict[str, Any]) -> str:
        return f"Surveying {self._rel(arguments)}"

    def _overview(self, base: Path) -> str:
        policy = self._policy
        files = list(policy.iter_files(base))
        by_ext: Counter[str] = Counter(p.suffix.lower() or "(none)" for p in files)
        top_dirs: Counter[str] = Counter()
        for p in files:
            rel = p.relative_to(base)
            top_dirs[rel.parts[0] if len(rel.parts) > 1 else "(root files)"] += 1

        stacks: list[str] = []
        for p in files:
            label = STACK_MARKERS.get(p.name)
            if label and label not in stacks and len(p.relative_to(base).parts) <= 3:
                stacks.append(label)

        hints: Counter[str] = Counter()
        for p in files:
            if p.suffix.lower() in SOURCE_EXTS and p.stat().st_size <= MAX_SCAN_BYTES:
                text = _read_small(policy, p)
                for label, pattern in API_HINTS.items():
                    n = len(pattern.findall(text))
                    if n:
                        hints[label] += n

        docs = [p for p in files if p.suffix.lower() in (".md", ".rst") and len(p.relative_to(base).parts) <= 2]
        api_files = [
            p for p in files
            if re.search(r"(openapi|swagger)[^/]*\.(json|ya?ml)$|\.postman_collection\.json$|api[^/]*\.md$", p.name, re.I)
        ]

        lines = [f"Project {policy.display(base)}  ({len(files)} readable files) {UNTRUSTED_NOTE}"]
        lines.append("Stack: " + (", ".join(stacks) if stacks else "unknown (no build/manifest files found)"))
        lines.append("File types: " + ", ".join(f"{ext} x{n}" for ext, n in by_ext.most_common(8)))
        lines.append("Top-level layout: " + ", ".join(f"{d} ({n})" for d, n in top_dirs.most_common(12)))
        if hints:
            lines.append("API code found: " + ", ".join(f"{label}: {n}" for label, n in hints.most_common()))
            scan = scan_project(policy, base)
            shown = [f"{e.method} /{e.path.strip('/')}" + (f" [{e.variant}]" if e.variant else "") for e in scan.endpoints[:MAX_ENDPOINT_LINES]]
            if shown:
                more = f" ... and {len(scan.endpoints) - len(shown)} more" if len(scan.endpoints) > len(shown) else ""
                lines.append(f"API endpoints ({len(scan.endpoints)}; base URL {scan.base_url}):\n  " + "\n  ".join(shown) + more)
        if api_files:
            lines.append("API docs/specs: " + ", ".join(policy.display(p) for p in api_files[:8]))
        if docs:
            lines.append("Docs: " + ", ".join(p.name for p in docs[:10]))

        pkg = next((p for p in files if p.name == "package.json" and p.parent == base), None)
        if pkg:
            try:
                data = json.loads(_read_small(policy, pkg) or "{}")
                deps = list((data.get("dependencies") or {}))[:12]
                lines.append(f"package.json: name={data.get('name')}; scripts={list((data.get('scripts') or {}))[:8]}; deps={deps}")
            except ValueError:
                pass
        gradle = next((p for p in files if p.name in ("build.gradle.kts", "build.gradle") and p.parent.name == "app"), None)
        if gradle:
            text = _read_small(policy, gradle)
            found = [w for w in ("retrofit", "room", "firebase", "hilt", "dagger", "compose", "coroutines", "glide", "okhttp") if w in text.lower()]
            app_id = re.search(r'applicationId\s*=?\s*"([^"]+)"', text)
            lines.append(f"Android app: applicationId={app_id.group(1) if app_id else '?'}; libraries: {', '.join(found)}")

        readme = next((p for p in docs if p.name.lower().startswith("readme")), None)
        if readme:
            head = redact(_read_small(policy, readme))[:500].strip()
            if head:
                lines.append("README start:\n" + head)
        lines.append("Next: read_file / code_outline for details, or generate_postman_collection for the API.")
        return "\n".join(lines)[:MAX_OVERVIEW_CHARS]

    async def execute(self, **arguments: Any) -> ToolResult:
        try:
            base = self._policy.resolve(arguments.get("path") if isinstance(arguments.get("path"), str) else "")
            if not base.is_dir():
                raise AccessDenied(f"'{self._policy.display(base)}' is not a directory.")
        except AccessDenied as exc:
            return ToolResult.failure(str(exc))
        return ToolResult.success(await asyncio.to_thread(self._overview, base))


# ------------------------------------------------------------------ outlines
_KT_JAVA = re.compile(
    r"^(?P<indent>\s*)(?:(?:public|private|protected|internal|open|abstract|final|data|sealed|static|override|suspend|inline|"
    r"companion|const|lateinit|actual|expect|value|inner|fun\s+interface)\s+)*"
    r"(?P<kind>class|interface|object|enum class|enum|fun|record|typealias)\s+(?:<[^>]+>\s*)?(?P<name>[\w.]+)"
)
_JAVA_METHOD = re.compile(
    r"^(?P<indent>\s*)(?:(?:public|private|protected|static|final|abstract|synchronized|native)\s+)+"
    r"[\w<>\[\],.? ]+?\s+(?P<name>\w+)\s*\([^;{]*\)\s*(?:throws [\w., ]+)?\s*\{?\s*$"
)
_PY = re.compile(r"^(?P<indent>\s*)(?P<kind>class|def|async def)\s+(?P<name>\w+)")
_JS = re.compile(
    r"^(?P<indent>\s*)(?:export\s+)?(?:default\s+)?(?:async\s+)?(?P<kind>function\*?|class|interface|type|enum)\s+(?P<name>\w+)"
)
_JS_ARROW = re.compile(
    r"^(?P<indent>\s*)(?:export\s+)?(?:const|let|var)\s+(?P<name>\w+)\s*(?::[^=]+)?=\s*(?:async\s*)?(?:\([^)]*\)|\w+)\s*(?::[^=]+)?=>"
)
_GO = re.compile(r"^(?P<kind>func|type)\s+(?:\([^)]*\)\s*)?(?P<name>\w+)")
_RUST = re.compile(r"^(?P<indent>\s*)(?:pub(?:\([^)]*\))?\s+)?(?P<kind>fn|struct|enum|trait|impl|mod)\s+(?P<name>\w+)")
_GENERIC = re.compile(r"^(?P<indent>\s*)(?P<kind>class|def|function|module|struct|protocol|extension)\s+(?P<name>\w+)")

_LANG = {
    ".kt": (_KT_JAVA,), ".kts": (_KT_JAVA,), ".scala": (_KT_JAVA,), ".swift": (_KT_JAVA, _GENERIC), ".dart": (_KT_JAVA, _GENERIC),
    ".java": (_KT_JAVA, _JAVA_METHOD), ".cs": (_KT_JAVA, _JAVA_METHOD),
    ".py": (_PY,), ".js": (_JS, _JS_ARROW), ".jsx": (_JS, _JS_ARROW), ".ts": (_JS, _JS_ARROW), ".tsx": (_JS, _JS_ARROW),
    ".go": (_GO,), ".rs": (_RUST,), ".php": (_GENERIC,), ".rb": (_GENERIC,),
}
MAX_SYMBOLS = 100


def outline(text: str, suffix: str) -> list[tuple[int, str, str, int]]:
    """(line, kind, name, indent) for the declarations in `text`. Regex-based: fast, approximate."""
    patterns = _LANG.get(suffix.lower())
    if not patterns:
        return []
    symbols = []
    for number, line in enumerate(text.splitlines(), 1):
        if len(line) > 300 or line.lstrip().startswith(("//", "#", "*", "/*")):
            continue
        for pattern in patterns:
            m = pattern.match(line)
            if m:
                groups = m.groupdict()
                kind = groups.get("kind") or "method"
                symbols.append((number, kind, groups["name"], len(groups.get("indent") or "")))
                break
        if len(symbols) >= MAX_SYMBOLS:
            break
    return symbols


class CodeOutlineTool(FileTool):
    name = "code_outline"
    description = (
        "List the classes, functions and other declarations in a source file (with line numbers), or a "
        "summary of every source file in a folder. Use it to find where things are before reading."
    )
    parameters = {
        "type": "object",
        "properties": {"path": {"type": "string", "description": "Source file or folder"}},
        "required": ["path"],
    }

    def describe(self, arguments: dict[str, Any]) -> str:
        return f"Outlining {self._rel(arguments)}"

    def _file_outline(self, path: Path) -> str:
        symbols = outline(self._policy.read_text(path), path.suffix)
        if not symbols:
            return f"{self._policy.display(path)}: no declarations recognised for '{path.suffix}' files."
        body = "\n".join(f"L{n} {'  ' * min(indent // 4, 3)}{kind} {name}" for n, kind, name, indent in symbols)
        cap = f"\n(first {MAX_SYMBOLS} shown)" if len(symbols) >= MAX_SYMBOLS else ""
        return f"Outline of {self._policy.display(path)} {UNTRUSTED_NOTE}:\n{body}{cap}"

    def _folder_outline(self, base: Path) -> str:
        rows = []
        for path in self._policy.iter_files(base):
            if path.suffix.lower() not in _LANG:
                continue
            try:
                symbols = outline(self._policy.read_text(path, max_bytes=MAX_SCAN_BYTES), path.suffix)
            except AccessDenied:
                continue
            tops = [name for _, kind, name, indent in symbols if indent == 0 and kind != "fun"][:4]
            rows.append(f"{self._policy.display(path)}: {len(symbols)} decl" + (f" [{', '.join(tops)}]" if tops else ""))
        if not rows:
            return f"No recognised source files under {self._policy.display(base)}."
        shown = rows[:70]
        more = f"\n... and {len(rows) - len(shown)} more files" if len(rows) > len(shown) else ""
        return f"Source files under {self._policy.display(base)} {UNTRUSTED_NOTE}:\n" + "\n".join(shown) + more

    async def execute(self, **arguments: Any) -> ToolResult:
        try:
            path = self._policy.resolve(arguments.get("path") if isinstance(arguments.get("path"), str) else "")
            text = await asyncio.to_thread(self._file_outline if path.is_file() else self._folder_outline, path)
        except AccessDenied as exc:
            return ToolResult.failure(str(exc))
        return ToolResult.success(text[:4500])
