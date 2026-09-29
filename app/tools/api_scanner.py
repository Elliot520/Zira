"""Find the HTTP API a project uses or exposes, from its source code and docs.

Extractors (all regex-based, so approximate but fast and dependency-free):
- Retrofit interfaces (Kotlin/Java), including the `@Url` style where the path lives in a constant and
  the request body type comes from the interface method signature;
- Express, FastAPI/Flask and Spring routes;
- OpenAPI/Swagger JSON files;
- Markdown API docs (tables with a method and a path column).
Results are merged: code supplies request-body examples, docs supply auth/status/descriptions.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from app.tools.filesystem import AccessDenied, FileAccessPolicy

METHODS = ("GET", "POST", "PUT", "DELETE", "PATCH", "HEAD", "OPTIONS")
MAX_FIELDS = 45
GENERIC_PREFIXES = {"api", "v1", "v2", "v3", "rest", "public"}
MAX_DEPTH = 3
NOT_LITERAL = object()

_ROUTE_LIKE = re.compile(r"^[A-Za-z][\w\-]*(?:/[\w\-{}:.]+)+/?$")
_MIME_FIRST = {"image", "application", "text", "audio", "video", "multipart", "font", "android", "java", "kotlin"}
_URLISH_NAME = re.compile(r"url|api|endpoint|webhook|route|path", re.I)


@dataclass
class Endpoint:
    method: str
    path: str
    name: str = ""
    auth: bool | None = None
    description: str = ""
    status: str = ""
    body: Any = None
    body_type: str = ""
    form: list[tuple[str, str, str]] | None = None  # (key, "text"|"file", example)
    query: dict[str, str] = field(default_factory=dict)
    sources: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    variant: str = ""
    from_code: bool = False

    @property
    def folder(self) -> str:
        """Postman folder: the resource segment, skipping generic prefixes like /api and /v1."""
        if self.path.startswith(("http://", "https://")):
            return "External"
        original = [seg for seg in self.path.strip("/").split("/") if seg]
        if len(original) < 2:
            return "General"
        segs = original
        while len(segs) > 1 and segs[0].lower() in GENERIC_PREFIXES:
            segs = segs[1:]
        first = segs[0]
        if first.startswith(("{", ":")):
            return "General"
        return first[:1].upper() + first[1:]


@dataclass
class ScanResult:
    endpoints: list[Endpoint]
    base_url: str
    stacks: list[str]
    warnings: list[str]
    docs_files: list[str] = field(default_factory=list)


def norm_path(path: str) -> str:
    return re.sub(r"\{\w+\}|:\w+", "{}", path.strip().strip("/").lower())


# ---------------------------------------------------------------- small parsers
def split_top_level(text: str, sep: str = ",") -> list[str]:
    parts, depth, cur, in_str = [], 0, [], ""
    for ch in text:
        if in_str:
            cur.append(ch)
            if ch == in_str:
                in_str = ""
            continue
        if ch in "\"'":
            in_str = ch
        elif ch in "<([{":
            depth += 1
        elif ch in ">)]}":
            depth = max(0, depth - 1)
        elif ch == sep and depth == 0:
            parts.append("".join(cur).strip())
            cur = []
            continue
        cur.append(ch)
    tail = "".join(cur).strip()
    if tail:
        parts.append(tail)
    return parts


def balanced(text: str, open_idx: int, open_ch: str = "(", close_ch: str = ")") -> tuple[str, int] | None:
    """Text inside the bracket at `open_idx`, and the index after the matching close."""
    depth, in_str = 0, ""
    for i in range(open_idx, len(text)):
        ch = text[i]
        if in_str:
            if ch == in_str and text[i - 1] != "\\":
                in_str = ""
            continue
        if ch == '"':
            in_str = ch
        elif ch == open_ch:
            depth += 1
        elif ch == close_ch:
            depth -= 1
            if depth == 0:
                return text[open_idx + 1 : i], i + 1
    return None


def split_generic(t: str) -> tuple[str, list[str]]:
    t = t.strip()
    m = re.match(r"^([\w.]+)\s*<(.*)>$", t)
    if not m:
        return t.split(".")[-1], []
    return m.group(1).split(".")[-1], split_top_level(m.group(2))


def parse_literal(default: str | None) -> Any:
    if default is None:
        return NOT_LITERAL
    d = default.strip().rstrip(",")
    d = re.sub(r"\s*//.*$", "", d).strip()
    if re.fullmatch(r'"[^"$\\]*"', d):
        return d[1:-1]
    if re.fullmatch(r"-?\d+[Ll]?", d):
        return int(d.rstrip("Ll"))
    if re.fullmatch(r"-?\d+\.\d*[fF]?|-?\.\d+[fF]?", d):
        return float(d.rstrip("fF"))
    if d in ("true", "false"):
        return d == "true"
    if d == "null":
        return None
    if re.fullmatch(r"(?:emptyList|arrayListOf|listOf|mutableListOf|ArrayList<[^>]*>|ArrayList)\(\)", d):
        return []
    if re.fullmatch(r"(?:emptyMap|mapOf|hashMapOf|mutableMapOf|HashMap<[^>]*>|HashMap)\(\)", d):
        return {}
    return NOT_LITERAL


# ---------------------------------------------------------------- model index
@dataclass
class Field_:
    name: str
    type: str
    default: Any = NOT_LITERAL


@dataclass
class ClassInfo:
    name: str
    fields: list[Field_] = field(default_factory=list)
    enum_values: list[str] = field(default_factory=list)


_KT_CLASS = re.compile(
    r"^[ \t]*(?:@[\w.]+(?:\([^)]*\))?[ \t]*)*(?:(?:public|private|internal|open|abstract|data|sealed|inner|annotation|enum)[ \t]+)*"
    r"class[ \t]+(?P<name>\w+)",
    re.M,
)
_ANNOT = re.compile(r"@(\w+)(?:\(([^)]*)\))?")
_KT_PARAM = re.compile(r"^(?P<ann>(?:@\w+(?:\([^)]*\))?\s*)*)(?:(?:private|override|internal)\s+)*(?:val|var)\s+(?P<n>\w+)\s*(?::\s*(?P<t>[^=]+?))?\s*(?:=\s*(?P<d>.+))?$", re.S)
_KT_PROP = re.compile(
    r"^(?P<indent>\s*)(?:(?:private|public|protected|internal|open|override|lateinit|const)\s+)*(?:val|var)\s+(?P<n>\w+)"
    r"\s*(?::\s*(?P<t>[^=\n]+?))?\s*(?:=\s*(?P<d>[^\n]+?))?\s*$"
)
_JAVA_FIELD = re.compile(
    r"^\s*(?:private|public|protected)\s+(?!static)(?:final\s+)?(?P<t>[\w<>\[\], ?.]+?)\s+(?P<n>\w+)\s*(?:=\s*(?P<d>[^;]+))?;"
)
_JAVA_CLASS = re.compile(r"^[ \t]*(?:public\s+|final\s+|abstract\s+)*class\s+(?P<name>\w+)", re.M)


def _alias(annotations: str) -> str | None:
    for name, args in _ANNOT.findall(annotations or ""):
        if name in ("SerializedName", "JsonProperty", "Json", "SerialName") and args:
            m = re.search(r'"([^"]+)"', args)
            if m:
                return m.group(1)
    return None


def _skip_annotations(annotations: str) -> bool:
    return any(name in ("Transient",) for name, _ in _ANNOT.findall(annotations or ""))


def _strip_strings(line: str) -> str:
    return re.sub(r'"(?:\\.|[^"\\])*"', '""', line)


class ModelIndex:
    """name -> ClassInfo, built from Kotlin/Java sources; used to turn @Body types into JSON examples."""

    def __init__(self) -> None:
        self.classes: dict[str, ClassInfo] = {}

    def add_source(self, text: str, suffix: str) -> None:
        if suffix == ".kt":
            self._add_kotlin(text)
        elif suffix == ".java":
            self._add_java(text)

    def _add_kotlin(self, text: str) -> None:
        for m in _KT_CLASS.finditer(text):
            name = m.group("name")
            if name in self.classes:
                continue
            info = ClassInfo(name)
            is_enum = re.search(r"\benum\s+class\b", text[m.start() : m.end()]) is not None
            pos = m.end()
            rest = text[pos:]
            gen = re.match(r"\s*<[^>]*>", rest)
            if gen:
                pos += gen.end()
            ctor = re.match(r"\s*(?:(?:private|internal|public)\s+)?(?:constructor\s*)?\(", text[pos : pos + 60])
            body_start = None
            if ctor:
                open_idx = pos + ctor.end() - 1
                got = balanced(text, open_idx)
                if got:
                    params, after = got
                    for p in split_top_level(params):
                        pm = _KT_PARAM.match(p.strip())
                        if pm and not _skip_annotations(pm.group("ann")):
                            info.fields.append(
                                Field_(_alias(pm.group("ann")) or pm.group("n"), (pm.group("t") or "").strip(), parse_literal(pm.group("d")))
                            )
                    pos = after
            brace = re.match(r"[^{\n]*\{", text[pos : pos + 200])
            if brace:
                body_start = pos + brace.end() - 1
            if body_start is not None:
                got = balanced(text, body_start, "{", "}")
                if got:
                    body = got[0]
                    if is_enum:
                        head = re.split(r";|\n\s*\n|\bfun\b|\bval\b|\bvar\b", body, maxsplit=1)[0]
                        info.enum_values = [v.split("(")[0].strip() for v in split_top_level(re.sub(r"//.*", "", head)) if re.match(r"^\w+", v.strip())]
                    else:
                        self._body_props(body, info)
            self.classes[name] = info

    @staticmethod
    def _body_props(body: str, info: ClassInfo) -> None:
        depth, pending = 0, ""
        lines = body.splitlines()
        for i, line in enumerate(lines):
            clean = _strip_strings(re.sub(r"//.*$", "", line))
            if depth == 0:
                stripped = line.strip()
                if stripped.startswith("@") and not re.search(r"\b(?:val|var|fun)\b", stripped):
                    pending += " " + stripped
                else:
                    m = _KT_PROP.match(line)
                    inline_ann = " ".join(a for a in re.findall(r"@\w+(?:\([^)]*\))?", line))
                    nxt = lines[i + 1].strip() if i + 1 < len(lines) else ""
                    computed = re.search(r"\bget\s*\(\)|\bby\b|\bset\s*\(", line) or nxt.startswith(("get()", "get ()"))
                    if m and not computed and not _skip_annotations(pending + " " + inline_ann):
                        t = (m.group("t") or "").strip()
                        default = parse_literal(m.group("d"))
                        if not t and default is not NOT_LITERAL:
                            t = "String" if isinstance(default, str) else "Boolean" if isinstance(default, bool) else "Int" if isinstance(default, int) else "Double"
                        info.fields.append(Field_(_alias(pending + " " + inline_ann) or m.group("n"), t, default))
                    pending = ""
            depth += clean.count("{") - clean.count("}")

    def _add_java(self, text: str) -> None:
        for m in _JAVA_CLASS.finditer(text):
            name = m.group("name")
            if name in self.classes:
                continue
            info = ClassInfo(name)
            brace = text.find("{", m.end())
            got = balanced(text, brace, "{", "}") if brace != -1 else None
            if got:
                depth, pending = 0, ""
                for line in got[0].splitlines():
                    clean = _strip_strings(line)
                    if depth == 0:
                        s = line.strip()
                        if s.startswith("@") and "(" in s and ";" not in s:
                            pending += " " + s
                        else:
                            fm = _JAVA_FIELD.match(line)
                            if fm and "transient" not in line:
                                info.fields.append(Field_(_alias(pending + " " + " ".join(re.findall(r"@\w+\([^)]*\)", line))) or fm.group("n"), fm.group("t").strip(), parse_literal(fm.group("d"))))
                            pending = ""
                    depth += clean.count("{") - clean.count("}")
            self.classes[name] = info

    # ------------------------------------------------------------- sampling
    def sample_type(self, t: str, depth: int = 0, seen: frozenset[str] = frozenset()) -> Any:
        base, args = split_generic(t.strip().rstrip("?").strip())
        if base in ("String", "CharSequence", "char", "Char"):
            return "string"
        if base in ("Int", "Long", "Short", "Byte", "Integer", "int", "long", "short", "BigInteger", "Number"):
            return 0
        if base in ("Double", "Float", "double", "float", "BigDecimal"):
            return 0.0
        if base in ("Boolean", "boolean"):
            return False
        if base in ("Date", "LocalDateTime", "LocalDate", "Instant", "Timestamp", "OffsetDateTime", "ZonedDateTime"):
            return "2024-01-01T00:00:00Z"
        if base in ("List", "ArrayList", "MutableList", "Set", "HashSet", "MutableSet", "Collection", "Array", "LinkedList", "Iterable"):
            inner = args[0] if args else (t[:-2] if t.endswith("[]") else "")
            return [self.sample_type(inner, depth + 1, seen)] if inner and depth < MAX_DEPTH else []
        if base.endswith("[]"):
            return [self.sample_type(base[:-2], depth + 1, seen)] if depth < MAX_DEPTH else []
        if base in ("Map", "HashMap", "MutableMap", "LinkedHashMap", "JsonObject"):
            return {}
        cls = self.classes.get(base)
        if cls:
            if cls.enum_values:
                return cls.enum_values[0]
            if depth < MAX_DEPTH and base not in seen:
                return self.sample_class(base, depth + 1, seen | {base})[0]
            return {}
        return None

    def sample_class(self, name: str, depth: int = 0, seen: frozenset[str] = frozenset()) -> tuple[dict[str, Any], bool]:
        """(example object, truncated?)"""
        cls = self.classes.get(name)
        if cls is None:
            return {}, False
        out: dict[str, Any] = {}
        for f in cls.fields[:MAX_FIELDS]:
            value = f.default if f.default is not NOT_LITERAL else self.sample_type(f.type, depth, seen | {name})
            out[f.name] = value
        return out, len(cls.fields) > MAX_FIELDS


# ------------------------------------------------------------------- reading
class ProjectReader:
    def __init__(self, policy: FileAccessPolicy, base: Path) -> None:
        self.policy, self.base = policy, base
        self._files = list(policy.iter_files(base))
        self._cache: dict[Path, str] = {}

    def files(self, *suffixes: str) -> list[Path]:
        return [p for p in self._files if p.suffix.lower() in suffixes]

    def named(self, *names: str) -> list[Path]:
        return [p for p in self._files if p.name in names]

    def text(self, path: Path) -> str:
        if path not in self._cache:
            try:
                self._cache[path] = self.policy.read_text(path, max_bytes=800_000)
            except AccessDenied:
                self._cache[path] = ""
        return self._cache[path]

    def rel(self, path: Path) -> str:
        return self.policy.display(path)


# ------------------------------------------------------------------- Retrofit
@dataclass
class Param:
    kind: str  # body|header|query|path|part|field|url
    key: str
    type: str


@dataclass
class RetrofitMethod:
    name: str
    http: str
    path: str | None
    params: list[Param]
    multipart: bool
    source: str

    @property
    def body_type(self) -> str:
        return next((p.type for p in self.params if p.kind == "body"), "")

    @property
    def has_auth(self) -> bool:
        return any(p.kind == "header" and p.key.lower() == "authorization" for p in self.params)


_HTTP_ANN = re.compile(r'@(GET|POST|PUT|DELETE|PATCH|HEAD)\b(?:\(\s*(?:value\s*=\s*)?"([^"]*)"[^)]*\))?')
_PARAM_KIND = {"Body": "body", "Header": "header", "Query": "query", "Path": "path", "Part": "part", "Field": "field", "Url": "url", "QueryMap": "query", "PartMap": "part"}


def _parse_param(text: str, java: bool) -> Param | None:
    anns = _ANNOT.findall(text)
    kind, key = None, ""
    for name, args in anns:
        if name in _PARAM_KIND:
            kind = _PARAM_KIND[name]
            m = re.search(r'"([^"]+)"', args or "")
            key = m.group(1) if m else ""
            break
    if kind is None:
        return None
    rest = _ANNOT.sub("", text).strip()
    if java:
        parts = rest.rsplit(None, 1)
        ptype, pname = (parts[0], parts[1]) if len(parts) == 2 else (rest, "")
    else:
        m = re.match(r"(\w+)\s*:\s*(.+)$", rest, re.S)
        pname, ptype = (m.group(1), m.group(2)) if m else ("", rest)
    return Param(kind, key or pname, ptype.strip().rstrip("?").strip())


def parse_retrofit_interface(text: str, source: str) -> list[RetrofitMethod]:
    java = source.endswith(".java")
    anns = list(_HTTP_ANN.finditer(text))
    methods = []
    for i, ann in enumerate(anns):
        limit = anns[i + 1].start() if i + 1 < len(anns) else len(text)
        window = text[ann.end() : limit]
        fm = re.search(r"\bfun\s+(\w+)\s*\(", window) if not java else re.search(r"\bCall<[^\n]*?>\s+(\w+)\s*\(", window)
        if not fm:
            continue
        open_idx = ann.end() + fm.end() - 1
        got = balanced(text, open_idx)
        if not got:
            continue
        params = [p for p in (_parse_param(x, java) for x in split_top_level(got[0])) if p]
        between = text[ann.end() : ann.end() + fm.start()]
        methods.append(RetrofitMethod(fm.group(1), ann.group(1), ann.group(2), params, "@Multipart" in between or "@Multipart" in text[max(0, ann.start() - 60) : ann.start()], source))
    return methods


def collect_constants(reader: ProjectReader) -> tuple[dict[str, str], dict[str, str]]:
    """(route-like constants, all string constants)."""
    routes: dict[str, str] = {}
    strings: dict[str, str] = {}
    pat = re.compile(r'(?:const\s+)?(?:val|var|static\s+final\s+String)\s+(\w+)\s*(?::\s*String)?\s*=\s*"([^"$\\\n]+)"')
    for path in reader.files(".kt", ".java"):
        for name, value in pat.findall(reader.text(path)):
            strings.setdefault(name, value)
            first = value.split("/")[0].lower()
            if _ROUTE_LIKE.match(value) and first not in _MIME_FIRST and not re.search(r"\.\w{2,4}$", value):
                routes.setdefault(name, value.strip("/"))
    return routes, strings


_COMMENT_OR_STRING = re.compile(r'"(?:\\.|[^"\\\n])*"|/\*.*?\*/|//[^\n]*', re.S)


def strip_comments(text: str) -> str:
    """Remove // and /* */ comments but leave string literals (which may contain //) intact."""
    return _COMMENT_OR_STRING.sub(lambda m: m.group(0) if m.group(0).startswith('"') else " ", text)


_FUN = re.compile(r"\bfun\s+(?:<[^>]+>\s*)?(?:[\w.]+\.)?(\w+)\s*\(")


def split_functions(text: str) -> list[str]:
    starts = [m.start() for m in _FUN.finditer(text)]
    if not starts:
        return [text]
    starts.append(len(text))
    return [text[a:b] for a, b in zip(starts, starts[1:])]


_TOKENISH = re.compile(r"token|bearer|auth", re.I)


def pick_overloads(group: list[RetrofitMethod], args: list[str]) -> list[RetrofitMethod]:
    """Among same-named Retrofit methods, the ones a call with these arguments can be."""
    same_count = [m for m in group if len(m.params) == len(args)]
    if len(same_count) <= 1:
        return same_count
    def score(m: RetrofitMethod) -> int:
        total = 0
        for i, prm in enumerate(m.params):
            looks_token = bool(_TOKENISH.search(args[i]))
            if prm.kind == "header":
                total += 1 if looks_token else 0
            elif prm.kind == "body":
                total += 0 if looks_token else 1
        return total
    best = max(score(m) for m in same_count)
    return [next(m for m in same_count if score(m) == best)]


def extract_retrofit(reader: ProjectReader, models: ModelIndex) -> tuple[list[Endpoint], list[str]]:
    warnings: list[str] = []
    interface_files = [p for p in reader.files(".kt", ".java") if "retrofit2.http" in reader.text(p)]
    methods: list[RetrofitMethod] = []
    for p in interface_files:
        methods.extend(parse_retrofit_interface(reader.text(p), reader.rel(p)))
    if not methods:
        return [], warnings
    routes, strings = collect_constants(reader)
    by_name: dict[str, list[RetrofitMethod]] = {}
    for m in methods:
        by_name.setdefault(m.name, []).append(m)

    endpoints: dict[tuple[str, str, str], Endpoint] = {}
    paired_consts: set[str] = set()
    called: set[str] = set()  # @Url methods paired with at least one URL
    seen_call: set[str] = set()  # methods that appear at a call site at all

    def resolve_all(expr: str, window: str = "") -> list[str]:
        """Every route `expr` can evaluate to: constants, BASE_URL + "lit", or BASE_URL + "prefix/" + if/when {"a"} else "b"."""
        found: list[str] = []
        for cname, cpath in routes.items():
            if re.search(rf"\b{re.escape(cname)}\b", expr):
                paired_consts.add(cname)
                found.append(cpath)
        for m in re.finditer(r'BASE_URL\s*\+\s*"([^"$]*)"(\s*\+)?', expr):
            literal, continues = m.group(1).strip("/"), bool(m.group(2))
            if not continues:
                if _ROUTE_LIKE.match(literal):
                    found.append(literal)
                continue
            tail = (window or expr)[(window or expr).find(m.group(0)) + len(m.group(0)):]
            tail = re.split(r"\n\s*\n|retrofitInterface|\.enqueue", tail, maxsplit=1)[0][:300]
            for lit in re.findall(r'"(\w+)"', tail)[:6]:
                joined = f"{literal}/{lit}".strip("/") if not m.group(1).endswith("/") else f"{m.group(1)}{lit}".strip("/")
                if _ROUTE_LIKE.match(joined):
                    found.append(joined)
        for m in re.finditer(r'"(https?://[^"$\s]+)"\s*(?:\+\s*""\s*)?(?:\+\s*([\w.]+))?', expr):
            suffix = "{{" + m.group(2).split(".")[-1] + "}}" if m.group(2) else ""
            found.append(m.group(1) + suffix)
        return list(dict.fromkeys(found))

    stripped: dict[Path, str] = {}

    def stripped_text(path: Path) -> str:
        if path not in stripped:
            stripped[path] = strip_comments(reader.text(path))
        return stripped[path]

    wrapper_cache: dict[tuple[str, int], list[str]] = {}

    def wrapper_targets(fname: str, index: int) -> list[str]:
        """Routes that callers pass as argument `index` of the wrapper function `fname`."""
        key = (fname, index)
        if key not in wrapper_cache:
            found: list[str] = []
            for p in reader.files(".kt", ".java"):
                text = stripped_text(p)
                for cm in re.finditer(rf"\b{re.escape(fname)}\s*\(", text):
                    if re.search(r"\bfun\s+(?:[\w<>.]+\.)?$", text[max(0, cm.start() - 40) : cm.start()]):
                        continue  # the definition itself
                    got = balanced(text, cm.end() - 1)
                    args = split_top_level(got[0]) if got else []
                    if index < len(args):
                        found += resolve_all(args[index])
            wrapper_cache[key] = list(dict.fromkeys(found))
        return wrapper_cache[key]

    def add(m: RetrofitMethod, path: str, chunk: str, variant: str = "", response_key: str = "") -> None:
        key = (m.http, norm_path(path), variant)
        ep = endpoints.get(key)
        if ep is None:
            ep = Endpoint(m.http, path if path.startswith("http") else path.strip("/"), from_code=True, variant=variant)
            ep.name = m.name
            if path.startswith("http"):
                ep.notes.append("External third-party API (not part of this project's own backend).")
            ep.auth = m.has_auth
            ep.sources.append(f"retrofit {m.source.split('/')[-1]}#{m.name}")
            ep.query = {p.key: "" for p in m.params if p.kind == "query"}
            endpoints[key] = ep
        if m.multipart:
            file_key = (re.search(r'createFormData\(\s*"(\w+)"', chunk) or [None, "file"])[1]
            ep.form = [
                (p.key, "file" if "MultipartBody.Part" in p.type or "File" in p.type else "text", "" if "Multipart" in p.type else "example")
                for p in m.params if p.kind in ("part", "field")
            ]
            ep.form = [(file_key if kind == "file" else k, kind, ex) for k, kind, ex in ep.form]
        bt = m.body_type
        if bt and ep.body is None and ep.form is None:
            base, args = split_generic(bt)
            ep.body_type = bt
            if base in ("Map", "HashMap", "MutableMap"):
                keys = re.findall(r'"(\w+)"\s+to\b|\.put\(\s*"(\w+)"|\["(\w+)"\]\s*=', chunk)
                ep.body = {k: "string" for group in keys for k in group if k}
            elif base in models.classes:
                ep.body, truncated = models.sample_class(base)
                if truncated:
                    ep.notes.append(f"{base} has more than {MAX_FIELDS} fields; only the first {MAX_FIELDS} are shown.")
            else:
                ep.body = models.sample_type(bt) if models.sample_type(bt) is not None else {}
        if variant and isinstance(ep.body, dict):
            ep.body["query_type"] = variant
            if response_key:
                ep.body["response_key"] = response_key

    for path in [p for p in reader.files(".kt", ".java") if p not in interface_files]:
        text = stripped_text(path)
        if not any(re.search(rf"\.{re.escape(n)}\(", text) for n in by_name):
            continue
        for chunk in split_functions(text):
            fn = _FUN.match(chunk)
            fn_params: list[str] = []
            if fn:
                got_params = balanced(chunk, fn.end() - 1)
                fn_params = [pm.group(1) for x in split_top_level(got_params[0]) if (pm := re.match(r"(?:vararg\s+)?(\w+)\s*:", x.strip()))] if got_params else []
            param_vars: dict[str, int] = {}
            url_vars: dict[str, list[str]] = {}
            for vm in re.finditer(r"(?:(?:val|var)\s+)?\b(\w+)\s*(?::\s*String)?\s*=(?!=)\s*([^\n]+)", chunk):
                for resolved in resolve_all(vm.group(2), chunk[vm.start(2):]):
                    url_vars.setdefault(vm.group(1), [])
                    if resolved not in url_vars[vm.group(1)]:
                        url_vars[vm.group(1)].append(resolved)
            for pv in re.finditer(r"\b(\w+)\s*(?::\s*String)?\s*=(?!=)\s*[^\n]*BASE_URL\s*\+\s*(\w+)\s*$", chunk, re.M):
                if pv.group(2) in fn_params:
                    param_vars[pv.group(1)] = fn_params.index(pv.group(2))
            variants = re.findall(r'query_type\s*=\s*(?:"(\w+)"|(?:\w+\.)?([A-Z][A-Z0-9_]+))', chunk)
            variant_values = [v or strings.get(c, "") for v, c in variants]
            response_keys = re.findall(r'response_key\s*=\s*"(\w+)"', chunk)
            for name, group in by_name.items():
                for call in re.finditer(rf"\.{re.escape(name)}\(", chunk):
                    got = balanced(chunk, call.end() - 1)
                    if not got:
                        continue
                    args = split_top_level(got[0])
                    if not args:
                        continue
                    candidates = pick_overloads(group, args)
                    if candidates:  # same name AND same argument count; other classes' methods can share the name
                        seen_call.add(name)
                    for m in candidates:
                        if m.path is not None:
                            add(m, m.path, chunk)
                            continue
                        first = args[0].strip()
                        targets = url_vars.get(first) or resolve_all(first)
                        if not targets and fn:
                            index = param_vars.get(first, fn_params.index(first) if first in fn_params else None)
                            if index is not None:
                                targets = wrapper_targets(fn.group(1), index)
                        if not targets:
                            continue
                        called.add(m.name)
                        for target in targets:
                            if variant_values:
                                for i, v in enumerate(dict.fromkeys(variant_values)):
                                    add(m, target, chunk, v, response_keys[i] if i < len(response_keys) else "")
                            else:
                                add(m, target, chunk)

    for m in methods:  # literal-path methods that have no call site still count
        if m.path is not None and (m.http, norm_path(m.path), "") not in endpoints:
            add(m, m.path, "")

    for cname, cpath in routes.items():  # URL constants that no call site used
        if cname not in paired_consts and _URLISH_NAME.search(cname) and not any(k[1] == norm_path(cpath) for k in endpoints):
            ep = Endpoint("POST", cpath, name=cname, from_code=True, sources=[f"constant {cname}"], notes=["No call site found in the code; the HTTP method and body are guesses."])
            endpoints[("POST", norm_path(cpath), "")] = ep

    url_methods = [m.name for m in methods if m.path is None]
    dead = sorted(n for n in dict.fromkeys(url_methods) if n not in seen_call)
    unresolved = sorted(n for n in dict.fromkeys(url_methods) if n in seen_call and n not in called)
    if dead:
        warnings.append(f"{len(dead)} Retrofit methods are never called in the code (unused): {', '.join(dead[:8])}")
    if unresolved:
        warnings.append(f"{len(unresolved)} Retrofit methods are called with a URL that could not be worked out: {', '.join(unresolved[:8])}")
    return list(endpoints.values()), warnings


# --------------------------------------------------------- Express / FastAPI / Spring
def extract_express(reader: ProjectReader) -> list[Endpoint]:
    route = re.compile(r"\b(?P<obj>router|app|\w*[Rr]outer)\.(?P<m>get|post|put|delete|patch)\(\s*(?P<q>['\"`])(?P<p>[^'\"`]+)(?P=q)")
    mount = re.compile(r"\bapp\.use\(\s*['\"`](?P<prefix>[^'\"`]+)['\"`]\s*,\s*(?P<var>\w+)")
    require = re.compile(r"(?:const|let|var|import)\s+(?P<var>\w+)\s*(?:=\s*require\(|from\s*)\s*['\"](?P<mod>[^'\"]+)['\"]")
    js = reader.files(".js", ".ts", ".mjs", ".cjs")
    prefixes: dict[str, str] = {}
    for p in js:
        text = reader.text(p)
        reqs = {m.group("var"): m.group("mod") for m in require.finditer(text)}
        for m in mount.finditer(text):
            mod = reqs.get(m.group("var"))
            if mod:
                prefixes[Path(mod).stem.split(".")[0].lower()] = m.group("prefix")
    out = []
    for p in js:
        text = reader.text(p)
        stem = p.stem.split(".")[0].lower()
        prefix = prefixes.get(stem, "")
        for m in route.finditer(text):
            full = "/".join(x.strip("/") for x in (prefix, m.group("p")) if x.strip("/"))
            body_keys = re.findall(r"req\.body\.(\w+)|const\s*\{([^}]+)\}\s*=\s*req\.body", text[m.end(): m.end() + 1500])
            keys = [k for a, b in body_keys for k in ([a] if a else [x.strip().split(":")[0] for x in b.split(",")]) if k]
            ep = Endpoint(m.group("m").upper(), full, name=full.split("/")[-1] or "root", from_code=True, sources=[f"express {reader.rel(p)}"])
            if m.group("m") != "get" and keys:
                ep.body = {k: "string" for k in dict.fromkeys(keys)}
            out.append(ep)
    return out


def _pydantic_sample(text: str, name: str, depth: int = 0) -> Any:
    m = re.search(rf"^class\s+{re.escape(name)}\s*\((?P<bases>[^)]*)\)\s*:\s*\n(?P<body>(?:[ \t]+.*\n?|\n)+)", text, re.M)
    if not m:
        return {}
    out: dict[str, Any] = {}
    for line in m.group("body").splitlines():
        fm = re.match(r"^\s{1,8}(?P<n>\w+)\s*:\s*(?P<t>[^=\n]+?)(?:\s*=\s*(?P<d>.+))?$", line)
        if not fm or fm.group("n").startswith("_") or "ClassVar" in fm.group("t"):
            continue
        t = fm.group("t").strip()
        d = fm.group("d")
        value: Any = NOT_LITERAL
        if d:
            ds = d.strip()
            if ds == "None":
                value = None
            elif ds in ("True", "False"):
                value = ds == "True"
            else:
                value = parse_literal(re.sub(r"^'([^'\\]*)'$", r'"\1"', ds))
        if value is NOT_LITERAL:
            inner = re.sub(r"\s*\|\s*None|Optional\[(.+)\]", r"\1", t)
            lower = inner.lower()
            value = ("string" if lower.startswith("str") else 0 if lower.startswith("int") else 0.0 if lower.startswith("float")
                     else False if lower.startswith("bool") else [] if lower.startswith(("list", "set", "tuple")) else {} if lower.startswith("dict") else None)
            if value is None and re.match(r"^[A-Z]\w+$", inner) and depth < MAX_DEPTH:
                value = _pydantic_sample(text, inner, depth + 1)
        out[fm.group("n")] = value
    return out


def extract_python_routes(reader: ProjectReader) -> list[Endpoint]:
    out = []
    for p in reader.files(".py"):
        text = reader.text(p)
        if not re.search(r"\b(FastAPI|APIRouter|Flask|Blueprint)\b", text):
            continue
        prefix_m = re.search(r"APIRouter\([^)]*prefix\s*=\s*['\"]([^'\"]+)['\"]", text)
        prefix = prefix_m.group(1) if prefix_m else ""
        for m in re.finditer(r"@(?P<obj>\w+)\.(?P<m>get|post|put|delete|patch)\(\s*['\"](?P<p>[^'\"]*)['\"](?P<rest>[^\n]*)\)\s*\n\s*(?:async\s+)?def\s+(?P<fn>\w+)\s*\((?P<args>[^)]*)\)", text):
            path = "/".join(x.strip("/") for x in (prefix, m.group("p")) if x.strip("/"))
            ep = Endpoint(m.group("m").upper(), path, name=m.group("fn"), from_code=True, sources=[f"fastapi {reader.rel(p)}#{m.group('fn')}"])
            for arg in split_top_level(m.group("args")):
                am = re.match(r"(\w+)\s*:\s*([A-Z]\w+)", arg.strip())
                if am and am.group(2) not in ("Request", "Response", "WebSocket", "BackgroundTasks", "Depends") and m.group("m") != "get":
                    ep.body, ep.body_type = _pydantic_sample(text, am.group(2)), am.group(2)
            out.append(ep)
        for m in re.finditer(r"@(?P<obj>\w+)\.route\(\s*['\"](?P<p>[^'\"]*)['\"](?:\s*,\s*methods\s*=\s*\[(?P<ms>[^\]]*)\])?\)\s*\n\s*def\s+(?P<fn>\w+)", text):
            for method in re.findall(r"['\"](\w+)['\"]", m.group("ms") or "'GET'"):
                out.append(Endpoint(method.upper(), m.group("p").strip("/"), name=m.group("fn"), from_code=True, sources=[f"flask {reader.rel(p)}#{m.group('fn')}"]))
    return out


def extract_spring(reader: ProjectReader, models: ModelIndex) -> list[Endpoint]:
    out = []
    for p in reader.files(".java", ".kt"):
        text = reader.text(p)
        if "Mapping" not in text or "@RestController" not in text and "@Controller" not in text:
            continue
        cm = re.search(r'@RequestMapping\(\s*(?:value\s*=\s*)?(?:path\s*=\s*)?\{?"([^"]*)"', text)
        prefix = cm.group(1) if cm else ""
        for m in re.finditer(r'@(?P<k>Get|Post|Put|Delete|Patch)Mapping(?:\(\s*(?:value\s*=\s*|path\s*=\s*)?\{?"(?P<p>[^"]*)"[^)]*\))?\s*(?:\n\s*)*(?:public\s+)?[\w<>?,\[\] ]*?\s*(?:fun\s+)?(?P<fn>\w+)\s*\((?P<args>[^)]*)\)', text):
            path = "/".join(x.strip("/") for x in (prefix, m.group("p") or "") if x.strip("/"))
            ep = Endpoint(m.group("k").upper(), path, name=m.group("fn"), from_code=True, sources=[f"spring {reader.rel(p)}#{m.group('fn')}"])
            bm = re.search(r"@RequestBody\s+(?:final\s+)?(?:(\w+)\s+\w+|\w+\s*:\s*(\w+))", m.group("args"))
            if bm:
                tname = bm.group(1) or bm.group(2)
                ep.body_type = tname
                ep.body = models.sample_class(tname)[0] if tname in models.classes else {}
            out.append(ep)
    return out


def extract_openapi(reader: ProjectReader) -> list[Endpoint]:
    out = []
    for p in reader.files(".json"):
        if not re.search(r"openapi|swagger", p.name, re.I):
            continue
        try:
            spec = json.loads(reader.text(p))
        except ValueError:
            continue
        if not isinstance(spec, dict) or "paths" not in spec:
            continue
        schemas = (spec.get("components") or {}).get("schemas") or spec.get("definitions") or {}

        def sample(schema: Any, depth: int = 0) -> Any:
            if not isinstance(schema, dict) or depth > MAX_DEPTH:
                return None
            if "$ref" in schema:
                return sample(schemas.get(schema["$ref"].split("/")[-1]), depth + 1)
            if "example" in schema:
                return schema["example"]
            t = schema.get("type")
            if t == "object" or "properties" in schema:
                return {k: sample(v, depth + 1) for k, v in list((schema.get("properties") or {}).items())[:MAX_FIELDS]}
            if t == "array":
                return [sample(schema.get("items"), depth + 1)]
            return {"string": "string", "integer": 0, "number": 0.0, "boolean": False}.get(t)

        for path, ops in (spec.get("paths") or {}).items():
            for method, op in (ops or {}).items():
                if method.upper() not in METHODS or not isinstance(op, dict):
                    continue
                ep = Endpoint(method.upper(), path.strip("/"), name=op.get("operationId") or path.strip("/").split("/")[-1],
                              description=op.get("summary") or op.get("description") or "", from_code=True,
                              sources=[f"openapi {reader.rel(p)}"])
                content = ((op.get("requestBody") or {}).get("content") or {}).get("application/json") or {}
                if content.get("schema") is not None:
                    ep.body = sample(content["schema"])
                ep.query = {q["name"]: "" for q in op.get("parameters", []) if isinstance(q, dict) and q.get("in") == "query" and "name" in q}
                if op.get("security"):
                    ep.auth = True
                out.append(ep)
    return out


# ------------------------------------------------------------------ Markdown docs
_STATUS_ICONS = {"✅": "active", "⚠️": "deprecated", "🔧": "backend/admin", "❌": "not mounted"}


def extract_markdown_docs(reader: ProjectReader) -> tuple[list[Endpoint], str, list[str]]:
    out: list[Endpoint] = []
    base_url = ""
    used: list[str] = []
    for p in reader.files(".md"):
        text = reader.text(p)
        if not re.search(r"^\|\s*(?:GET|POST|PUT|DELETE|PATCH)\s*\|", text, re.M | re.I):
            continue
        used.append(reader.rel(p))
        bm = re.search(r"\*\*Base URL:?\*\*:?\s*`([^`]+)`", text)
        if bm and not base_url:
            base_url = bm.group(1).strip()
        columns: list[str] = []
        for line in text.splitlines():
            if not line.startswith("|"):
                continue
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            if cells and cells[0].lower() == "method":
                columns = [c.lower() for c in cells]
                continue
            if not columns or len(cells) < 2 or cells[0].upper() not in METHODS:
                continue
            row = dict(zip(columns, cells))
            path = re.sub(r"[`*]", "", row.get("endpoint", cells[1])).strip().strip("/")
            if not path or " " in path:
                continue
            status_cell = row.get("status", "")
            status = next((label for icon, label in _STATUS_ICONS.items() if icon in status_cell), "")
            auth_cell = row.get("auth", "").lower()
            ep = Endpoint(cells[0].upper(), path, name=path.split("/")[-1], description=row.get("description", ""),
                          status=status, sources=[f"docs {p.name}"])
            if auth_cell.startswith("yes"):
                ep.auth = True
            elif auth_cell.startswith("no"):
                ep.auth = False
            out.append(ep)
        # "Known query_type values" table -> variants of the generic endpoint(s)
        m = re.search(r"### Known `query_type` values[^\n]*\((?:`)?([\w/]+)(?:`)?\)", text)
        for qm in re.finditer(r"^\|\s*`(\w+)`(?:\s*/\s*`(\w+)`)?\s*\|\s*(?:`(\w+)`|—|-)\s*\|([^|\n]*)\|", text, re.M):
            for qt in filter(None, (qm.group(1), qm.group(2))):
                if qt == "query_type":  # the table's own header row
                    continue
                ep = Endpoint("POST", (m.group(1) if m else "data/genericApiData"), name=qt, variant=qt, sources=[f"docs {p.name}"],
                              status="active" if "Active" in qm.group(4) else "pending backend" if "Pending" in qm.group(4) else "",
                              body={"query_type": qt, **({"response_key": qm.group(3)} if qm.group(3) else {})})
                out.append(ep)
    return out, base_url, used


# ------------------------------------------------------------------ orchestration
def detect_base_url(reader: ProjectReader) -> str:
    for p in reader.named("build.gradle.kts", "build.gradle"):
        m = re.search(r'BASE_URL["\']?\s*,\s*["\']?\\?"(https?://[^"\\\']+)', reader.text(p))
        if m:
            return m.group(1)
    for p in reader.files(".kt", ".java"):
        m = re.search(r'\bBASE_URL\s*(?::\s*String)?\s*=\s*"(https?://[^"]+)"', reader.text(p))
        if m:
            return m.group(1)
    for p in reader.files(".js", ".ts", ".mjs"):
        m = re.search(r"\.listen\(\s*(?:process\.env\.PORT\s*\|\|\s*)?(\d{4,5})", reader.text(p))
        if m:
            return f"http://localhost:{m.group(1)}"
    for p in reader.files(".py"):
        if re.search(r"\bFastAPI\(", reader.text(p)):
            return "http://localhost:8000"
    return "http://localhost:8080"


def merge(groups: list[list[Endpoint]]) -> list[Endpoint]:
    merged: dict[tuple[str, str, str], Endpoint] = {}
    for group in groups:
        for ep in group:
            key = (ep.method, norm_path(ep.path), ep.variant)
            cur = merged.get(key)
            if cur is None:
                merged[key] = ep
                continue
            cur.sources.extend(s for s in ep.sources if s not in cur.sources)
            cur.notes.extend(n for n in ep.notes if n not in cur.notes)
            cur.description = cur.description or ep.description
            cur.status = cur.status or ep.status
            if ep.auth is not None and (ep.status or cur.auth is None):
                cur.auth = ep.auth  # docs are the server's word on auth; code only knows what the app sends
            if cur.body is None and cur.form is None:
                cur.body, cur.body_type, cur.form = ep.body, ep.body_type, ep.form
            cur.query = cur.query or ep.query
            cur.from_code = cur.from_code or ep.from_code
    return _resolve_method_conflicts(list(merged.values()))


def _resolve_method_conflicts(endpoints: list[Endpoint]) -> list[Endpoint]:
    """Retrofit `@Url` helpers often have a GET and a POST overload, so the same URL can look like both.
    For URLs known ONLY from Retrofit call sites (not documented, not a REST framework route) keep POST
    and drop the GET, noting it. REST frameworks and docs legitimately have GET and POST on one path."""
    def retrofit_only(e: Endpoint) -> bool:
        return bool(e.sources) and all(src.startswith(("retrofit", "constant")) for src in e.sources)

    by_path: dict[str, list[Endpoint]] = {}
    for ep in endpoints:
        if retrofit_only(ep):
            by_path.setdefault(norm_path(ep.path), []).append(ep)
    drop: set[int] = set()
    for group in by_path.values():
        posts = [e for e in group if e.method == "POST"]
        gets = [e for e in group if e.method == "GET"]
        if posts and gets:
            for e in gets:
                drop.add(id(e))
                posts[0].notes.append("The code also calls this URL with GET.")
    return [e for e in endpoints if id(e) not in drop]


def scan_project(policy: FileAccessPolicy, base: Path) -> ScanResult:
    reader = ProjectReader(policy, base)
    models = ModelIndex()
    for p in reader.files(".kt", ".java"):
        models.add_source(reader.text(p), p.suffix.lower())

    warnings: list[str] = []
    stacks: list[str] = []
    groups: list[list[Endpoint]] = []

    retrofit, w = extract_retrofit(reader, models)
    warnings += w
    if retrofit:
        stacks.append("Retrofit (Android/Kotlin/Java client)")
    groups.append(retrofit)
    for label, eps in (
        ("Express", extract_express(reader)),
        ("FastAPI/Flask", extract_python_routes(reader)),
        ("Spring", extract_spring(reader, models)),
        ("OpenAPI", extract_openapi(reader)),
    ):
        if eps:
            stacks.append(label)
        groups.append(eps)
    docs, doc_base, doc_files = extract_markdown_docs(reader)
    if docs:
        stacks.append("Markdown API docs")
    groups.append(docs)

    endpoints = merge(groups)
    base_url = (doc_base or detect_base_url(reader)).rstrip("/")
    only_docs = [e for e in endpoints if not e.from_code]
    if only_docs:
        warnings.append(f"{len(only_docs)} endpoints come only from the docs, so their request bodies are empty placeholders.")
    if not endpoints:
        warnings.append("No API endpoints were found. Supported: Retrofit, Express, FastAPI/Flask, Spring, OpenAPI JSON, Markdown API tables.")
    return ScanResult(endpoints, base_url, stacks, warnings, doc_files)
