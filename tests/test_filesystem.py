"""Sandboxed read-only file access: containment, secret blocking, redaction and the read tools."""

from __future__ import annotations

import os

import pytest

from app.tools.filesystem import (
    AccessDenied,
    FileAccessPolicy,
    ListDirectoryTool,
    ReadFileTool,
    SearchFilesTool,
    is_denied_file,
    redact,
)


@pytest.fixture
def project(tmp_path):
    root = tmp_path / "proj"
    (root / "src").mkdir(parents=True)
    (root / "node_modules").mkdir()
    (root / ".git").mkdir()
    (root / "src" / "App.kt").write_text('class App {\n  fun hello() = "hi"\n}\n')
    (root / "src" / "SecretsManager.kt").write_text("class SecretsManager\n")
    (root / "README.md").write_text("# Demo project\n")
    (root / ".env").write_text("API_KEY=super-secret-value-123\n")
    (root / ".env.example").write_text("API_KEY=\n")
    (root / "keystore.properties").write_text("storePassword=hunter2hunter2\n")
    (root / "secrets.txt").write_text("nope\n")
    (root / "node_modules" / "dep.js").write_text("function dep() {}\n")
    (root / ".git" / "config").write_text("[core]\n")
    (root / "blob.bin").write_bytes(b"\x00\x01\x02binary")
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "passwd.txt").write_text("root:x:0:0\n")
    os.symlink(outside / "passwd.txt", root / "src" / "link.txt")
    os.symlink(outside, root / "linkdir")
    return root


@pytest.fixture
def policy(project):
    return FileAccessPolicy([project])


# --------------------------------------------------------------- containment
def test_resolves_paths_inside_root(policy, project):
    assert policy.resolve(str(project / "src" / "App.kt")) == (project / "src" / "App.kt").resolve()
    assert policy.resolve("src/App.kt") == (project / "src" / "App.kt").resolve()  # relative to the root


@pytest.mark.parametrize(
    "raw",
    ["/etc/passwd", "../outside/passwd.txt", "src/../../outside/passwd.txt", "~", "/", ""],
)
def test_rejects_paths_outside_root(policy, raw):
    with pytest.raises(AccessDenied):
        policy.resolve(raw)


def test_rejects_symlink_escaping_the_root(policy, project):
    with pytest.raises(AccessDenied):
        policy.resolve(str(project / "src" / "link.txt"))
    with pytest.raises(AccessDenied):
        policy.resolve(str(project / "linkdir" / "passwd.txt"))


def test_sibling_folder_with_same_prefix_is_not_inside(tmp_path, project):
    sibling = tmp_path / "proj-evil"
    sibling.mkdir()
    (sibling / "x.txt").write_text("x")
    with pytest.raises(AccessDenied):
        FileAccessPolicy([project]).resolve(str(sibling / "x.txt"))


def test_disabled_without_roots():
    assert not FileAccessPolicy([]).enabled


# ----------------------------------------------------------- secret blocking
@pytest.mark.parametrize(
    "name",
    [".env", ".env.production", "server.pem", "release.keystore", "ziraago.jks", "keystore.properties",
     "local.properties", "gradle.properties", "google-services.json", "id_rsa", "id_rsa.pub", ".npmrc",
     ".mcp.json", "secrets.txt", "my-credentials.json", "prod.tfvars", "ServiceAccount-x.json"],
)
def test_secret_files_are_denied(name):
    assert is_denied_file(name)


@pytest.mark.parametrize(
    "name", [".env.example", "keystore.properties.example", "SecretsManager.kt", "CredentialStore.java", "app.py", "README.md"]
)
def test_normal_and_template_files_are_allowed(name):
    assert not is_denied_file(name)


def test_policy_refuses_secret_files_and_git_dir(policy, project):
    for raw in (".env", "keystore.properties", "secrets.txt", ".git/config"):
        with pytest.raises(AccessDenied):
            policy.resolve(raw)
    assert policy.resolve(".env.example")


def test_iter_files_skips_deps_secrets_git_and_escaping_symlinks(policy, project):
    names = {p.name for p in policy.iter_files(project)}
    assert {"App.kt", "README.md", "SecretsManager.kt", ".env.example"} <= names
    assert names.isdisjoint({".env", "keystore.properties", "secrets.txt", "dep.js", "config", "link.txt", "passwd.txt"})


# ------------------------------------------------------------------ redaction
@pytest.mark.parametrize(
    "text,secret",
    [
        ('val apiKey = "AIzaSyA1234567890abcdefghijklmnopqrstuv"', "AIzaSyA1234567890abcdefghijklmnopqrstuv"),
        ('password = "correct-horse-battery"', "correct-horse-battery"),
        ('"client_secret": "abcd1234efgh5678"', "abcd1234efgh5678"),
        ("key: rzp_live_AbCdEf123456", "rzp_live_AbCdEf123456"),
        ("AKIAIOSFODNN7EXAMPLE", "AKIAIOSFODNN7EXAMPLE"),
        ("token eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.abcdefghijklmnop", "eyJhbGciOiJIUzI1NiJ9"),
        ("-----BEGIN PRIVATE KEY-----\nMIIEvQ\n-----END PRIVATE KEY-----", "MIIEvQ"),
    ],
)
def test_redact_hides_credentials(text, secret):
    out = redact(text)
    assert secret not in out and "redacted" in out


def test_redact_leaves_ordinary_code_alone():
    code = "var password: String? = null\nfun checkPassword(password: String): Boolean = true"
    assert redact(code) == code


# ------------------------------------------------------------------ the tools
async def test_list_directory(policy, project):
    result = await ListDirectoryTool(policy).execute(path=str(project))
    assert result.ok
    out = result.output
    assert "src/" in out and "README.md" in out and ".env.example" in out
    assert "node_modules/  (skipped" in out
    assert ".env\n" not in out and "keystore.properties" not in out and "secrets.txt" not in out
    assert "secret/system entries hidden" in out


async def test_list_directory_without_path_shows_allowed_roots(policy, project):
    result = await ListDirectoryTool(policy).execute()
    assert str(project.resolve()) in result.output


async def test_list_directory_rejects_outside_and_files(policy, project):
    assert not (await ListDirectoryTool(policy).execute(path="/etc")).ok
    assert not (await ListDirectoryTool(policy).execute(path=str(project / "README.md"))).ok


async def test_read_file_returns_content_with_untrusted_label(policy, project):
    result = await ReadFileTool(policy).execute(path=str(project / "src" / "App.kt"))
    assert result.ok
    assert "fun hello()" in result.output and "untrusted" in result.output and "lines 1-3 of 3" in result.output


async def test_read_file_is_chunked_with_continuation_hint(policy, project):
    big = project / "big.txt"
    big.write_text("\n".join(f"line {i} " + "x" * 40 for i in range(400)))
    tool = ReadFileTool(policy, max_chars=1000)
    first = await tool.execute(path=str(big))
    assert "[more: call read_file with start_line=" in first.output
    next_line = int(first.output.rsplit("start_line=", 1)[1].split("]")[0])
    second = await tool.execute(path=str(big), start_line=next_line)
    assert f"line {next_line - 1} " in second.output  # 1-based: continues right where it stopped


async def test_read_file_redacts_secrets_in_allowed_files(policy, project):
    (project / "Config.kt").write_text('val key = "AIzaSyA1234567890abcdefghijklmnopqrstuv"\n')
    result = await ReadFileTool(policy).execute(path=str(project / "Config.kt"))
    assert "AIzaSy" not in result.output and "redacted" in result.output


@pytest.mark.parametrize("raw", [".env", "keystore.properties", "/etc/passwd", "../outside/passwd.txt", "blob.bin", "src", "missing.txt"])
async def test_read_file_refuses(policy, raw):
    result = await ReadFileTool(policy).execute(path=raw)
    assert not result.ok and result.error


async def test_search_files(policy, project):
    result = await SearchFilesTool(policy).execute(query="HELLO")
    assert result.ok and "App.kt:2:" in result.output
    assert "dep.js" not in result.output


async def test_search_never_reaches_secret_or_dependency_files(policy, project):
    for query in ("super-secret-value", "hunter2hunter2", "function dep"):
        assert "No matches" in (await SearchFilesTool(policy).execute(query=query)).output


async def test_search_glob_and_path(policy, project):
    (project / "src" / "Other.java").write_text("class Other { void hello() {} }\n")
    kt = await SearchFilesTool(policy).execute(query="hello", glob="*.kt")
    assert "App.kt" in kt.output and "Other.java" not in kt.output
    scoped = await SearchFilesTool(policy).execute(query="Demo", path=str(project / "src"))
    assert "No matches" in scoped.output


async def test_search_validates_input(policy):
    assert not (await SearchFilesTool(policy).execute(query="a")).ok
    assert not (await SearchFilesTool(policy).execute()).ok
    assert not (await SearchFilesTool(policy).execute(query="hello", path="/etc")).ok


async def test_search_caps_matches(policy, project):
    (project / "many.txt").write_text("\n".join("needle here" for _ in range(200)))
    result = await SearchFilesTool(policy).execute(query="needle")
    assert result.output.count("many.txt:") == 30 and "first 30 shown" in result.output


def test_file_tools_are_private_data_readers_and_only_relevant_for_project_talk(policy):
    tool = ReadFileTool(policy)
    assert tool.reads_private_data and not tool.sends_data_out
    assert tool.relevant("read /Users/me/proj/README.md")
    assert tool.relevant("make a postman collection for my project")
    assert not tool.relevant("what is the capital of France")
    assert not tool.relevant("hello there")
